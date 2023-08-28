"""
Additional utilities to work with the CTU-13 dataset.
"""

import sys
import numpy as np
import pandas as pd

from typing import List, Tuple
from multiprocessing import Pool
from collections import OrderedDict

from netpois.ctu13_botnet_detection.src import config_neris
from netpois.ctu13_botnet_detection.src.aggregated_features_bro_logs import (
    construct_header,
)


def find_internal_ip(tc: pd.DataFrame):
    INTERNAL = "147.32."
    src_ips = tc["id.orig_h"].to_numpy().astype(str)
    dst_ips = tc["id.resp_h"].to_numpy().astype(str)
    assert src_ips.shape == dst_ips.shape

    # Find positions of src_ips where the ip starts with INTERNAL
    src_pos = np.where([ip.startswith(INTERNAL) for ip in src_ips])[0]
    dst_pos = np.where([ip.startswith(INTERNAL) for ip in dst_ips])[0]

    # Compute intersection of src_pos and dst_pos
    int_to_int = np.intersect1d(src_pos, dst_pos)

    internal_ips = np.full_like(src_ips, "")
    internal_ips[dst_pos] = dst_ips[dst_pos]
    internal_ips[src_pos] = src_ips[src_pos]

    missing = np.where(internal_ips == "")[0]

    missing_and_int_to_int = np.union1d(int_to_int, missing)
    # Delete the entries corresponding to missing ips and
    # internal-to-internal connections
    internal_ips = np.delete(internal_ips, missing_and_int_to_int)

    assert internal_ips.shape[0] == src_ips.shape[0] - missing_and_int_to_int.shape[0]

    return internal_ips, missing, int_to_int


def vectorize_by_time(
    train_conn: pd.DataFrame, test_conn: pd.DataFrame, window_size: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_conn = train_conn.sort_values(by=["ts"])
    test_conn = test_conn.sort_values(by=["ts"])

    train_conn_labels = train_conn["label"]
    test_conn_labels = test_conn["label"]
    train_conn.drop(columns=["label"], inplace=True)
    test_conn.drop(columns=["label"], inplace=True)
    train_conn.drop(columns=["ts"], inplace=True)
    test_conn.drop(columns=["ts"], inplace=True)

    # Now we can split the connections into groups of size 100
    train_conn_groups = [
        train_conn.iloc[i : i + window_size]
        for i in range(0, train_conn.shape[0], window_size)
    ]
    test_conn_groups = [
        test_conn.iloc[i : i + window_size]
        for i in range(0, test_conn.shape[0], window_size)
    ]
    train_conn_labels_groups = [
        train_conn_labels.iloc[i : i + window_size]
        for i in range(0, train_conn_labels.shape[0], window_size)
    ]
    test_conn_labels_groups = [
        test_conn_labels.iloc[i : i + window_size]
        for i in range(0, test_conn_labels.shape[0], window_size)
    ]

    trn_y = np.array([np.max(l) for l in train_conn_labels_groups])
    tst_y = np.array([np.max(l) for l in test_conn_labels_groups])

    # Now we can concatenate each row in the group into a single row
    train_conn_groups = [g.to_numpy().reshape(-1) for g in train_conn_groups]
    test_conn_groups = [g.to_numpy().reshape(-1) for g in test_conn_groups]
    # Pad the last group with zeros if needed
    train_conn_groups[-1] = np.pad(
        train_conn_groups[-1],
        (0, train_conn_groups[-2].shape[0] - train_conn_groups[-1].shape[0]),
        "constant",
        constant_values=0,
    )
    test_conn_groups[-1] = np.pad(
        test_conn_groups[-1],
        (0, test_conn_groups[-2].shape[0] - test_conn_groups[-1].shape[0]),
        "constant",
        constant_values=0,
    )
    trn_x = np.array(train_conn_groups)
    tst_x = np.array(test_conn_groups)

    return trn_x, trn_y, tst_x, tst_y


def vectorize_by_IP_conn(
    conn: pd.DataFrame,
    w_size: int = 100,
    silent: bool = False,
) -> Tuple[np.ndarray, np.ndarray, list, np.ndarray]:

    # Add a column representing the original position in the dataframe
    conn["orig_row"] = np.arange(conn.shape[0])

    # Find internal IPs
    int_ips, missing, int_to_int = find_internal_ip(conn)
    missing_and_int_to_int = np.union1d(int_to_int, missing)
    # Remove rows where there is no internal IP
    conn = conn.drop(conn.index[missing_and_int_to_int])

    # Add internal IP column and remove original IP columns
    conn["internal_ip"] = int_ips
    conn.drop(columns=["id.orig_h", "id.resp_h"], inplace=True)

    if not silent:
        print("Train conn shape: {}".format(conn.shape))

    trn_unq_int_ip, trn_uniq_ip_conns = np.unique(
        conn["internal_ip"], return_counts=True
    )
    if not silent:
        print("Number of unique internal IPs: {}".format(trn_unq_int_ip.shape[0]))
        print("Mean of connections by unique IP: {}".format(np.mean(trn_uniq_ip_conns)))
        print(
            "Median of connections by unique IP: {}".format(
                np.median(trn_uniq_ip_conns)
            )
        )
        print("Max of connections by unique IP: {}".format(np.max(trn_uniq_ip_conns)))
        print("Min of connections by unique IP: {}".format(np.min(trn_uniq_ip_conns)))
        print("Std of connections by unique IP: {}".format(np.std(trn_uniq_ip_conns)))
        print("Var of connections by unique IP: {}".format(np.var(trn_uniq_ip_conns)))
        print(
            "Number of IPs with less than {} connections: {}".format(
                w_size, np.sum(trn_uniq_ip_conns < w_size)
            )
        )
        print(
            "Number of IPs with more than {} connections: {}".format(
                w_size, np.sum(trn_uniq_ip_conns >= w_size)
            )
        )

    conn = conn.sort_values(by=["internal_ip", "ts"])

    conn_grps, conn_lbl_grps, conn_row_grps = [], [], []
    for _, g in conn.groupby("internal_ip"):
        g_y = g["label"]
        g_rows = g["orig_row"]
        g.drop(columns=["label", "internal_ip", "ts", "orig_row"], inplace=True)
        conn_grps += [g.iloc[i : i + w_size] for i in range(0, g.shape[0], w_size)]
        conn_lbl_grps += [
            g_y.iloc[i : i + w_size] for i in range(0, g.shape[0], w_size)
        ]
        conn_row_grps += [
            g_rows.iloc[i : i + w_size] for i in range(0, g.shape[0], w_size)
        ]

    y = np.array([np.max(l) for l in conn_lbl_grps])
    rows = [g.to_numpy() for g in conn_row_grps]

    # Now we can concatenate each row in the group into a single row
    conn_grps = [g.to_numpy().reshape(-1) for g in conn_grps]

    # Needed to get to the correct shape
    conn.drop(columns=["label", "internal_ip", "ts", "orig_row"], inplace=True)
    n_cols = conn.shape[1]
    cols = conn.columns.to_numpy()

    # Pad every group to the same length
    for i in range(len(conn_grps)):
        conn_grps[i] = np.pad(
            conn_grps[i],
            (0, w_size * n_cols - conn_grps[i].shape[0]),
            "constant",
            constant_values=0,
        )

    x = np.array(conn_grps)

    return x, y, rows, cols


def vectorize_by_IP(
    train_conn: pd.DataFrame, test_conn: pd.DataFrame, window_size: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Find internal IPs
    train_int_ips, train_missing, train_int_to_int = find_internal_ip(train_conn)
    test_int_ips, test_missing, test_int_to_int = find_internal_ip(test_conn)

    train_missing_and_int_to_int = np.union1d(train_missing, train_int_to_int)
    test_missing_and_int_to_int = np.union1d(test_missing, test_int_to_int)

    # Remove rows where there is no internal IP
    train_conn = train_conn.drop(train_conn.index[train_missing_and_int_to_int])
    test_conn = test_conn.drop(test_conn.index[test_missing_and_int_to_int])

    # Add internal IP column and remove original IP columns
    train_conn["internal_ip"] = train_int_ips
    test_conn["internal_ip"] = test_int_ips
    train_conn.drop(columns=["id.orig_h", "id.resp_h"], inplace=True)
    test_conn.drop(columns=["id.orig_h", "id.resp_h"], inplace=True)

    print("Train conn shape: {}".format(train_conn.shape))
    print("Test conn shape: {}".format(test_conn.shape))

    trn_unq_int_ip, trn_uniq_ip_conns = np.unique(
        train_conn["internal_ip"], return_counts=True
    )
    print("Number of unique internal IPs: {}".format(trn_unq_int_ip.shape[0]))
    print("Mean of connections by unique IP: {}".format(np.mean(trn_uniq_ip_conns)))
    print("Median of connections by unique IP: {}".format(np.median(trn_uniq_ip_conns)))
    print("Max of connections by unique IP: {}".format(np.max(trn_uniq_ip_conns)))
    print("Min of connections by unique IP: {}".format(np.min(trn_uniq_ip_conns)))
    print("Std of connections by unique IP: {}".format(np.std(trn_uniq_ip_conns)))
    print("Var of connections by unique IP: {}".format(np.var(trn_uniq_ip_conns)))
    print(
        "Number of IPs with less than {} connections: {}".format(
            window_size, np.sum(trn_uniq_ip_conns < window_size)
        )
    )
    print(
        "Number of IPs with more than {} connections: {}".format(
            window_size, np.sum(trn_uniq_ip_conns >= window_size)
        )
    )

    train_conn = train_conn.sort_values(by=["internal_ip", "ts"])
    test_conn = test_conn.sort_values(by=["internal_ip", "ts"])

    (
        train_conn_groups,
        test_conn_groups,
        train_conn_labels_groups,
        test_conn_labels_groups,
    ) = ([], [], [], [])
    for _, g in train_conn.groupby("internal_ip"):
        g_y = g["label"]
        g.drop(columns=["label", "internal_ip", "ts"], inplace=True)
        train_conn_groups += [
            g.iloc[i : i + window_size] for i in range(0, g.shape[0], window_size)
        ]
        train_conn_labels_groups += [
            g_y.iloc[i : i + window_size] for i in range(0, g.shape[0], window_size)
        ]

    for _, g in test_conn.groupby("internal_ip"):
        g_y = g["label"]
        g.drop(columns=["label", "internal_ip", "ts"], inplace=True)
        test_conn_groups += [
            g.iloc[i : i + window_size] for i in range(0, g.shape[0], window_size)
        ]
        test_conn_labels_groups += [
            g_y.iloc[i : i + window_size] for i in range(0, g.shape[0], window_size)
        ]

    trn_y = np.array([np.max(l) for l in train_conn_labels_groups])
    tst_y = np.array([np.max(l) for l in test_conn_labels_groups])

    # Now we can concatenate each row in the group into a single row
    train_conn_groups = [g.to_numpy().reshape(-1) for g in train_conn_groups]
    test_conn_groups = [g.to_numpy().reshape(-1) for g in test_conn_groups]

    # Needed to get to the correct shape
    train_conn.drop(columns=["label", "internal_ip", "ts"], inplace=True)
    test_conn.drop(columns=["label", "internal_ip", "ts"], inplace=True)

    # Pad every group to the same length
    for i in range(len(train_conn_groups)):
        train_conn_groups[i] = np.pad(
            train_conn_groups[i],
            (0, window_size * train_conn.shape[1] - train_conn_groups[i].shape[0]),
            "constant",
            constant_values=0,
        )

    for i in range(len(test_conn_groups)):
        test_conn_groups[i] = np.pad(
            test_conn_groups[i],
            (0, window_size * test_conn.shape[1] - test_conn_groups[i].shape[0]),
            "constant",
            constant_values=0,
        )

    trn_x = np.array(train_conn_groups)
    tst_x = np.array(test_conn_groups)

    return trn_x, trn_y, tst_x, tst_y



def ports_to_known(resp_ports: np.ndarray) -> np.ndarray:
    """Replace ports with a known port if they are not in the list of known ports

    Args:
        resp_ports (np.ndarray): array of ports
        known_ports (List[int]): list of known ports

    Returns:
        np.ndarray: array of ports
    """
    OTHER_PORT = -1
    KNOWN_PORTS = [
        1,
        3,
        8,
        10,
        21,
        22,
        25,
        53,
        80,
        110,
        123,
        135,
        138,
        161,
        443,
        445,
        993,
        OTHER_PORT,
    ]

    return np.where(np.isin(resp_ports, KNOWN_PORTS), resp_ports, OTHER_PORT)


def label_conn_log(identifier: str, cl: pd.DataFrame) -> pd.DataFrame:
    """Add label column to dataframe from connection log

    Args:
        identifier (str): identifier of the capture
        cl (pd.DataFrame): connection log dataframe

    Returns:
        pd.DataFrame: dataframe with label column
    """
    src_ip = cl["id.orig_h"]
    dst_ip = cl["id.resp_h"]
    assert len(src_ip) == len(dst_ip)
    n_entries = len(src_ip)
    set_ips = set(config_neris.BOTNET_IPS[identifier])

    labels = [
        1 if src_ip.iloc[i] in set_ips or dst_ip.iloc[i] in set_ips else 0
        for i in range(n_entries)
    ]

    cl["label"] = labels
    return cl


def read_conn_log(
    filepath: str, allowed_fields: List[str] = None
) -> Tuple[OrderedDict, List[str]]:
    """Read data from a Zeek conn.log file

    Args:
        filepath (str): path to the conn.log file
        allowed_fields (List[str], optional): list of fields to extraxct. Defaults to None.

    Returns:
        Tuple[OrderedDict, List[str], Lis[str]]: dictionary of values for the file,
        header of the file
    """

    content = OrderedDict()

    # Save the file header so the file can be re-created
    file_header = []
    content["data"] = []
    content[
        "separator"
    ] = "\t"  # Set a default separator in case we don't get the separator

    with open(filepath) as infile:
        for line in infile.readlines():
            line = line.strip()

            # The first line is generally of the form: #separator SEP
            if line.startswith("#separator"):
                key = str(line[1:].split(" ")[0])
                value = str.encode(line[1:].split(" ")[1].strip()).decode(
                    "unicode_escape"
                )
                content[key] = value
                file_header.append(line)

            # Header lines start with a # and are of the form: #keySEPvalue(s)
            elif line.startswith("#"):
                key = str(line[1:].split(content.get("separator"))[0])
                value = line[1:].split(content.get("separator"))[1:]
                content[key] = value
                file_header.append(line)

            # Data lines are of the form: value(s)SEPvalue(s)
            else:
                data = line.split(content.get("separator"))

                # Check that the number of values in the row matches the number of fields
                if len(data) is len(content.get("fields")):
                    record = OrderedDict()

                    for x in range(0, len(data) - 1):

                        # If no fields are specified, or the current field is in the list of
                        # fields to include, add it to the record
                        if (
                            allowed_fields is None
                            or content.get("fields")[x] in allowed_fields
                        ):
                            record[content.get("fields")[x]] = data[x]

                        else:
                            raise Exception(
                                "Skipping field: {}".format(content.get("fields")[x])
                            )

                    content["data"].append(record)

                # Arrays are not the same length
                else:
                    raise Exception(
                        "The number of fields in the line does not match the number of fields in the header\n",
                        content.get("fields"),
                        "\n",
                        line,
                    )

    return content, file_header


# #######################################################
# From aggregated_features_bro_log.py


def new_values(cur_value, min, max, sum):
    if cur_value < min:
        min = cur_value
    if cur_value > max:
        max = cur_value
    return [sum + cur_value, min, max]


def aggregate_feats_for_subset(subset: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Mimic `aggregated_features_bro_log.py` for a subset DF

    Assumes the subset corresponds to a single window from a single IP.

    Args:
        subset (pd.DataFrame): subset data frame

    Returns:
        pd.DataFrame: extracted aggregated features
    """
    intip = config_neris.INTERNAL
    OTHER_PORT = -1
    KNOWN_PORTS = [
        1,
        3,
        8,
        10,
        21,
        22,
        25,
        53,
        80,
        110,
        123,
        135,
        138,
        161,
        443,
        445,
        993,
        OTHER_PORT,
    ]
    STAT_FEATURES_NUM = 31

    aggregate_t = None
    dict_for_window = {}

    for index, row in subset.iterrows():
        row = [0 if elem == "-" else elem for elem in row]
        (
            ts,
            uid,
            src_ip,
            src_port,
            dst_ip,
            dst_port,
            protocol,
            service,
            duration,
            bytes_outgoing,
            bytes_incoming,
            state,
            packets_outgoing,
            packets_incoming,
        ) = row
        ts, duration = float(ts), float(duration)
        (
            src_port,
            dst_port,
            bytes_outgoing,
            bytes_incoming,
            packets_outgoing,
            packets_incoming,
        ) = (
            int(src_port),
            int(dst_port),
            int(bytes_outgoing),
            int(bytes_incoming),
            int(packets_outgoing),
            int(packets_incoming),
        )

        if aggregate_t is None:
            aggregate_t = int(int(ts) // window * window)
        cur_aggregate_t = int(int(ts) // window * window)

        if cur_aggregate_t != aggregate_t:  # time window changed
            raise Exception("New window: {} - {}".format(cur_aggregate_t, aggregate_t))
        if not src_ip.startswith(intip) and not dst_ip.startswith(intip):
            raise Exception("No internal IP: {} - {}".format(src_ip, dst_ip))

        if src_ip.startswith(intip):
            internal_ip = src_ip
            external_ip = dst_ip
            features_index = 0
        else:
            internal_ip = dst_ip
            external_ip = src_ip
            features_index = 1

        orig_dst_port = dst_port  # keep the original value

        # replace the port number if it falls in OTHER
        if dst_port not in KNOWN_PORTS:
            dst_port = OTHER_PORT
        if internal_ip not in dict_for_window:
            dict_for_window[internal_ip] = OrderedDict()
            for port in KNOWN_PORTS:
                dict_for_window[internal_ip][port] = [
                    [0] * STAT_FEATURES_NUM,
                    [0] * STAT_FEATURES_NUM,
                    set(),
                    set(),
                    set(),
                    set(),
                    set(),
                    set(),
                ]

        flist = [
            bytes_in_sum,
            bytes_in_min,
            bytes_in_max,
            bytes_out_sum,
            bytes_out_min,
            bytes_out_max,
            pkts_in_sum,
            pkts_in_min,
            pkts_in_max,
            pkts_out_sum,
            pkts_out_min,
            pkts_out_max,
            duration_sum,
            duration_min,
            duration_max,
            tcp_count,
            udp_count,
            icmp_count,
            state_S0,
            state_S1,
            state_SF,
            state_REJ,
            state_S2,
            state_S3,
            state_RSTO,
            state_RSTR,
            state_RSTOS0,
            state_RSTRH,
            state_SH,
            state_SHR,
            state_OTH,
        ] = dict_for_window[internal_ip][dst_port][features_index]

        if all(v == 0 for v in flist):
            bytes_in_min = sys.maxsize
            bytes_out_min = sys.maxsize
            pkts_in_min = sys.maxsize
            pkts_out_min = sys.maxsize
            duration_min = sys.maxsize

        dict_for_window[internal_ip][dst_port][features_index] = (
            new_values(bytes_incoming, bytes_in_min, bytes_in_max, bytes_in_sum)
            + new_values(bytes_outgoing, bytes_out_min, bytes_out_max, bytes_out_sum)
            + new_values(packets_incoming, pkts_in_min, pkts_in_max, pkts_in_sum)
            + new_values(packets_outgoing, pkts_out_min, pkts_out_max, pkts_out_sum)
            + new_values(duration, duration_min, duration_max, duration_sum)
            + [
                tcp_count + int(protocol == "tcp"),
                udp_count + int(protocol == "udp"),
                icmp_count + int(protocol == "icmp"),
            ]
            + [
                state_S0 + int(state == "S0"),
                state_S1 + int(state == "S1"),
                state_SF + int(state == "SF"),
                state_REJ + int(state == "REJ"),
                state_S2 + int(state == "S2"),
                state_S3 + int(state == "S3"),
                state_RSTO + int(state == "RSTO"),
                state_RSTR + int(state == "RSTR"),
                state_RSTOS0 + int(state == "RSTOS0"),
                state_RSTRH + int(state == "RSTRH"),
                state_SH + int(state == "SH"),
                state_SHR + int(state == "SHR"),
                state_OTH + int(state == "OTH"),
            ]
        )

        assert (
            len(dict_for_window[internal_ip][dst_port][features_index])
            == STAT_FEATURES_NUM
        )

        if (
            external_ip
            not in dict_for_window[internal_ip][dst_port][features_index + 2]
        ):
            dict_for_window[internal_ip][dst_port][features_index + 2].add(external_ip)
        if src_port not in dict_for_window[internal_ip][dst_port][features_index + 4]:
            dict_for_window[internal_ip][dst_port][features_index + 4].add(src_port)
        if (
            orig_dst_port
            not in dict_for_window[internal_ip][dst_port][features_index + 6]
        ):
            dict_for_window[internal_ip][dst_port][features_index + 6].add(
                orig_dst_port
            )

    output_line = []
    cols = construct_header()[:-4]  # remove the last column
    for ip_i in dict_for_window:
        for port_j in dict_for_window[ip_i]:
            features = dict_for_window[ip_i][port_j]
            assert len(features[0]) == STAT_FEATURES_NUM == len(features[1])

            output_line.extend(features[0])
            output_line.append(len(features[2]))
            output_line.append(len(features[4]))
            output_line.append(len(features[6]))
            output_line.extend(features[1])
            output_line.append(len(features[3]))
            output_line.append(len(features[5]))
            output_line.append(len(features[7]))

    columns_mask = [
        True
        if not x.startswith("distinct_src_port")
        and not x.startswith("distinct_dst_port")
        else False
        for x in cols
    ]
    cols = np.array(cols)[columns_mask]
    output_line = np.array(output_line)[columns_mask]

    to_df = pd.DataFrame.from_dict(dict(zip(cols, output_line)), orient="index")

    return to_df
