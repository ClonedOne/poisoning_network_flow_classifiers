import math
import copy
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import defaultdict
from IPython.display import display
from typing import Tuple, List, Dict
from sklearn.neighbors import KernelDensity
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import norm, truncnorm, dweibull
from sklearn.metrics import f1_score, accuracy_score

from poisnet import ctu_utils, data_utils, constants
from poisnet.poison import shapbdr_feature_selectors


def visual_compare_closest(
    identifier: str,
    sel_features: np.ndarray,
    columns: np.ndarray,
    x: np.ndarray,
    assignment: np.ndarray,
    dists: np.ndarray,
):
    """Visually compare the chosen assignment to the closest observed point.

    Args:
        identifier (str): identifier of the value selection strategy
        sel_features (np.ndarray): indices of selected features
        columns (np.ndarray): original feature names
        x (np.ndarray): nontarget data
        assignment (np.ndarray): assignment values
        dists (np.ndarray): pairwise distances between the assignment and the nontarget data
    """

    print(
        "For the {} assignment, the closest sample has the following values:".format(
            identifier
        )
    )
    to_show = []
    for i, f in enumerate(sel_features):
        to_show.append(
            {
                "feature": f,
                "column": columns[f],
                "assignment": assignment[i],
                "observed": x[dists.argmin()][f],
            }
        )
    to_show = pd.DataFrame(to_show)
    display(to_show)


def find_prototype(
    x_trn: np.ndarray,
    x_tst: np.ndarray,
    y_trn: np.ndarray,
    y_tst: np.ndarray,
    x_nontarget: np.ndarray,
    tgt_cass: int,
    sel_features: np.ndarray,
    dists: np.ndarray,
) -> Tuple[np.ndarray, int, str]:

    # Position at which the train data has the same values as the chosen
    # assignment for the selected features
    ref_data = "train"
    ref_x = x_trn
    ref_y = y_trn
    d_amin = dists.argmin()
    pos_proto = np.where(
        (x_trn[:, sel_features] == x_nontarget[:, sel_features][d_amin]).all(axis=1)
        & (y_trn != tgt_cass)
    )[0]
    if len(pos_proto) == 0:
        pos_proto = np.where(
            (x_tst[:, sel_features] == x_nontarget[:, sel_features][d_amin]).all(axis=1)
            & (y_tst != tgt_cass)
        )[0]
        ref_data = "test"
        ref_x = x_tst
        ref_y = y_tst

    pos_proto = pos_proto[0]

    proto = ref_x[pos_proto]

    print("Considering {} point: {}".format(ref_data, pos_proto))
    print("Label of the point: {}".format(ref_y[pos_proto]))

    return proto, pos_proto, ref_data


def load_row_information(
    trn_csvs: List[str],
    tst_csvs: List[str],
    x_trn: np.ndarray,
    x_tst: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:

    aggr_feats_trn = []
    aggr_feats_tst = []
    for csv in trn_csvs:
        aggr_feats_trn.append(pd.read_csv(csv))
    for csv in tst_csvs:
        aggr_feats_tst.append(pd.read_csv(csv))

    aggr_feats_trn_sizes = [df.shape[0] for df in aggr_feats_trn]
    aggr_feats_tst_sizes = [df.shape[0] for df in aggr_feats_tst]

    # Only get the row numbers column
    x_trn_rows = np.concatenate([i.values[:, -1] for i in aggr_feats_trn])
    x_tst_rows = np.concatenate([i.values[:, -1] for i in aggr_feats_tst])

    del aggr_feats_trn, aggr_feats_tst

    assert x_trn.shape[0] == x_trn_rows.shape[0], "{} != {}".format(
        x_trn.shape[0], x_trn_rows.shape[0]
    )
    assert x_tst.shape[0] == x_tst_rows.shape[0], "{} != {}".format(
        x_tst.shape[0], x_tst_rows.shape[0]
    )

    return x_trn_rows, x_tst_rows, aggr_feats_trn_sizes, aggr_feats_tst_sizes


def find_included_feats(
    columns: np.ndarray, verbose: bool = False, exclude_cts: bool = False
) -> np.ndarray:
    """Identify the features to include in the attack.

    The structure of this function is designed to allow verbose visul output

    Args:
        columns (np.ndarray): Original feature names
        verbose (bool, optional): verbosity. Defaults to False.

    Returns:
        np.ndarray: Mask of the features to include in the attack
    """
    ports = sorted(set([i.split("_")[-1] for i in columns]))
    n_ports = len(ports)

    feats_mask = [True if ports[0] == i.split("_")[-1] else False for i in columns]
    feats_per_port = [
        i[: -(len(ports[0]) + 1)] for i, j in zip(columns, feats_mask) if j
    ]
    feats_per_port = sorted(feats_per_port)

    # Exclude all the features that contain the word "min" or "duration"
    exclude_strings = ["min", "duration"]
    if exclude_cts:
        exclude_strings = ["min", "duration", "count", "state", "distinct"]
    excluded_features = [
        i for i in feats_per_port if any([j in i for j in exclude_strings])
    ]

    # This is a mask for the features that we want to use for the attack
    included_features_mask = list(
        map(lambda x: not any([i in x for i in excluded_features]), columns)
    )
    included_features_mask = np.array(included_features_mask)

    assert included_features_mask.shape == columns.shape

    if verbose:
        print("\nTotal number of ports: {}\nPorts:\n{}".format(n_ports, ports))

        print(
            "\nCurrently using the following {} features for each port".format(
                len(feats_per_port)
            )
        )
        for i in range(0, math.ceil(len(feats_per_port) // 4), 1):
            print(
                "{:25}\t{:25}\t{:25}\t{:25}".format(
                    *feats_per_port[i * 4 : (i + 1) * 4]
                )
            )

        print("\nList of excluded features: {}".format(excluded_features))

        print(
            "\nNumber of features used for the attack: {} / {}\n".format(
                sum(included_features_mask), included_features_mask.shape
            )
        )

    return included_features_mask


def first_true(iterable, default=False, pred=None):
    """Returns the first true value in the iterable.

    If no true value is found, returns *default*

    If *pred* is not None, returns the first item
    for which pred(item) is true.

    """
    return next(filter(pred, iterable), default)


def find_internal_ip_in_subset(subset: np.ndarray) -> Tuple[str, str]:
    prefix = constants.internal_prefix[constants.neris_tag]
    origins = subset[:, 2]
    ip = first_true(origins, default="", pred=lambda x: x.startswith(prefix))
    if ip:
        return ip, "origin"
    destinations = subset[:, 4]
    ip = first_true(destinations, default="", pred=lambda x: x.startswith(prefix))
    if ip:
        return ip, "destination"
    raise ValueError("No internal IP found in subset")


def find_all_internal_ip_in_subset(
    subset: np.ndarray, col_orig_ip: int = 2, col_dest_ip: int = 4, prefix: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find all the occurrences of an internal IP in a subset
    This method assumes internal-internal/external-external traffic is not
    present in the subset

    Args:
        subset (np.ndarray): Numpy array of the conn log subset (pandas.values)
        col_orig_ip (int, optional): Column index of the origin IP. Defaults to 2.
        col_dest_ip (int, optional): Column index of the destination IP. Defaults to 4.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Internal IPs found, masks of positions of the
            internal IP in the subset for the origin and destination columns
    """
    if prefix is None:
        prefix = constants.internal_prefix[constants.neris_tag]


    origins = subset[:, col_orig_ip]
    destinations = subset[:, col_dest_ip]

    # Find all positions where the origin or destination is internal
    origins_mask = np.array([i.startswith(prefix) for i in origins])
    destinations_mask = np.array([i.startswith(prefix) for i in destinations])

    # If there are no internal IPs in the subset, raise an error
    if not any(origins_mask) and not any(destinations_mask):
        raise ValueError("No internal IP found in subset")

    int_ips = np.full(origins_mask.shape, "", dtype=object)
    int_ips[origins_mask] = subset[:, col_orig_ip][origins_mask]
    int_ips[destinations_mask] = subset[:, col_dest_ip][destinations_mask]
    if "" in int_ips:
        raise ValueError("There is a row with no internal IP")

    return int_ips, origins_mask, destinations_mask


def inject_poison(
    orig_rows: Dict[int, List[int]],
    orig_csv: pd.DataFrame,
    trigger: pd.DataFrame,
    trigger_window: int,
    trigger_orig_mask: np.ndarray,
    trigger_dest_mask: np.ndarray,
    window: int = 30,
    col_orig_ip: int = 2,
    col_dest_ip: int = 4,
) -> Tuple[pd.DataFrame, int]:

    injections = 0
    vals = orig_csv.values.copy()
    num_inject = len(orig_rows)
    trig_vals = trigger.values
    trigger_len = trig_vals.shape[0]

    num_new_rows = trigger_len * num_inject
    new_data = np.empty((vals.shape[0] + num_new_rows, vals.shape[1]), dtype=vals.dtype)

    start_idx = 0
    tot_injected_rows = 0
    injection_positions = []

    # Create a dictionary to map the original row indices of each point to the
    # respective first row. So we can scan them in order when injecting the poison
    lls = {}
    for l in orig_rows.values():
        lls[l[0]] = l

    for k in tqdm(sorted(lls), desc="Injecting poison", total=num_inject):
        rows = lls[k]

        # Identify the subset of the original dataset corresponding to
        # the point to poison
        subset = vals[rows]

        # Find which is the internal IP in the connection events of
        # the subset, and whether it is the origin or destination
        data_internal_ip, _ = find_internal_ip_in_subset(subset)

        # Find the time window of the subset
        data_window = int(int(subset[:, 0][0]) // window * window)

        # Adapt the trigger to the current point's window
        injection = trig_vals.copy()
        injection[:, col_orig_ip][trigger_orig_mask] = data_internal_ip
        injection[:, col_dest_ip][trigger_dest_mask] = data_internal_ip
        injection[:, 0] -= trigger_window
        injection[:, 0] += data_window

        # Insert the trigger into the data at the correct position.
        # This will be the first row of the original rows of the point
        injection_pos = rows[0] + (injections * trigger_len)
        injection_positions.append(injection_pos)
        new_data[start_idx:injection_pos] = vals[
            start_idx - tot_injected_rows : injection_pos - tot_injected_rows
        ]
        new_data[injection_pos : injection_pos + trigger_len] = injection

        start_idx = injection_pos + trigger_len
        tot_injected_rows += trigger_len
        injections += 1

        # del injection, subset, data_internal_ip, data_window

    assert tot_injected_rows == num_new_rows
    new_data[start_idx:] = vals[start_idx - tot_injected_rows :]
    # print(sorted(injection_positions))

    new_data = pd.DataFrame(new_data, columns=orig_csv.columns)
    return new_data, injections


def inject_poison_OLD(
    orig_rows: Dict[int, List[int]],
    orig_csv: pd.DataFrame,
    trigger: pd.DataFrame,
    trigger_window: int,
    trigger_orig_mask: np.ndarray,
    trigger_dest_mask: np.ndarray,
    window: int = 30,
    col_orig_ip: int = 2,
    col_dest_ip: int = 4,
) -> Tuple[pd.DataFrame, int]:

    injections = 0
    vals = orig_csv.copy().values
    trigger_len = trigger.shape[0]

    for _, rows in tqdm(
        orig_rows.items(), desc="Injecting poison", total=len(orig_rows)
    ):

        # Identify the subset of the original dataset corresponding to
        # the point to poison
        subset = orig_csv.values[rows]

        # Find which is the internal IP in the connection events of
        # the subset, and whether it is the origin or destination
        data_internal_ip, _ = find_internal_ip_in_subset(subset)

        # Find the time window of the subset
        data_window = int(int(subset[:, 0][0]) // window * window)

        # Adapt the trigger to the current point's window
        injection = trigger.copy().values
        injection[:, col_orig_ip][trigger_orig_mask] = data_internal_ip
        injection[:, col_dest_ip][trigger_dest_mask] = data_internal_ip
        injection[:, 0] -= trigger_window
        injection[:, 0] += data_window

        # Insert the trigger into the data at the correct position.
        # This will be the first row of the original rows of the point
        injection_pos = rows[0] + (injections * trigger_len)
        vals = np.concatenate(
            (vals[:injection_pos], injection, vals[injection_pos:]), axis=0
        )
        injections += 1

        del injection, subset, data_internal_ip, data_window

    vals = pd.DataFrame(vals, columns=orig_csv.columns)
    return vals, injections


def inject_poison_windowed_test(
    orig_rows: Dict[int, List[int]],
    orig_csv: pd.DataFrame,
    trigger: pd.DataFrame,
    trigger_orig_mask: np.ndarray,
    trigger_dest_mask: np.ndarray,
    window_n: int = 100.0,
    col_orig_ip: int = 2,
    col_dest_ip: int = 4,
) -> Tuple[pd.DataFrame, int]:

    injections = 0
    vals = orig_csv.copy().values
    trigger_len = trigger.shape[0]

    for _, rows in tqdm(
        orig_rows.items(), desc="Injecting poison", total=len(orig_rows)
    ):

        subset = orig_csv.values[rows]

        # Find which is the internal IP in the connection events of
        # the subset, and whether it is the origin or destination
        subset_int_ip, _ = find_internal_ip_in_subset(subset)

        # Adapt the trigger to the current point's window
        injection = trigger.copy().values
        injection[:, col_orig_ip][trigger_orig_mask] = subset_int_ip
        injection[:, col_dest_ip][trigger_dest_mask] = subset_int_ip

        # Set to fixed value (0) for fixed position trigger
        rand_offset = np.random.randint(0, min(rows.shape[0], window_n - trigger_len))

        # Set the timestamp to be the one at the injection point
        injection[:, 0] = subset[:, 0][rand_offset]

        # Insert the trigger into the data at the correct position.
        # This will be the first row of the original rows of the point
        # plus the number of injections times the length of the trigger
        # plus a randomized offset
        injection_pos = rows[0] + (injections * trigger_len) + rand_offset
        vals = np.concatenate(
            (vals[:injection_pos], injection, vals[injection_pos:]), axis=0
        )
        injections += 1

        del injection, subset, subset_int_ip

    vals = pd.DataFrame(vals, columns=orig_csv.columns)
    return vals, injections

def inject_poison_windowed(
    orig_rows: Dict[int, List[int]],
    orig_csv: pd.DataFrame,
    trigger: pd.DataFrame,
    trigger_orig_mask: np.ndarray,
    trigger_dest_mask: np.ndarray,
    window_n: int = 100.0,
    col_orig_ip: int = 2,
    col_dest_ip: int = 4,
) -> Tuple[pd.DataFrame, int]:

    injections = 0
    vals = orig_csv.copy().values
    trigger_len = trigger.shape[0]

    for _, rows in tqdm(
        orig_rows.items(), desc="Injecting poison", total=len(orig_rows)
    ):

        subset = orig_csv.values[rows]

        # Find which is the internal IP in the connection events of
        # the subset, and whether it is the origin or destination
        subset_int_ip, _ = find_internal_ip_in_subset(subset)

        # Adapt the trigger to the current point's window
        injection = trigger.copy().values
        injection[:, col_orig_ip][trigger_orig_mask] = subset_int_ip
        injection[:, col_dest_ip][trigger_dest_mask] = subset_int_ip

        # Set to fixed value (0) for fixed position trigger
        rand_offset = np.random.randint(0, min(rows.shape[0], window_n - trigger_len))

        # Set the timestamp to be the one at the injection point
        injection[:, 0] = injection[:, 0][rand_offset]

        # Insert the trigger into the data at the correct position.
        # This will be the first row of the original rows of the point
        # plus the number of injections times the length of the trigger
        # plus a randomized offset
        injection_pos = rows[0] + (injections * trigger_len) + rand_offset
        vals = np.concatenate(
            (vals[:injection_pos], injection, vals[injection_pos:]), axis=0
        )
        injections += 1

        del injection, subset, subset_int_ip

    vals = pd.DataFrame(vals, columns=orig_csv.columns)
    return vals, injections


def inject_trigger(
    orig_rows: Dict[int, List[int]],
    orig_csv: pd.DataFrame,
    trigger: pd.DataFrame,
    trigger_window: int,
    trigger_orig_mask: np.ndarray,
    trigger_dest_mask: np.ndarray,
    window: int = 30,
    col_orig_ip: int = 2,
    col_dest_ip: int = 4,
) -> Tuple[pd.DataFrame, int]:

    injections = 0
    vals = orig_csv.values.copy()
    num_inject = len(orig_rows)
    trig_vals = trigger.values
    trigger_len = trig_vals.shape[0]

    # num_new_rows = trigger_len * num_inject
    new_data = []

    # Create a dictionary to map the original row indices of each point to the
    # respective first row. So we can scan them in order when injecting the poison
    lls = {}
    for l in orig_rows.values():
        lls[l[0]] = l

    for k in tqdm(sorted(lls), desc="Injecting poison", total=num_inject):
        rows = lls[k]

        # Identify the subset of the original dataset corresponding to
        # the point to poison
        subset = vals[rows]

        # Find which is the internal IP in the connection events of
        # the subset, and whether it is the origin or destination
        data_internal_ip, _ = find_internal_ip_in_subset(subset)

        # Find the time window of the subset
        data_window = int(int(subset[:, 0][0]) // window * window)

        # Adapt the trigger to the current point's window
        injection = trig_vals.copy()
        injection[:, col_orig_ip][trigger_orig_mask] = data_internal_ip
        injection[:, col_dest_ip][trigger_dest_mask] = data_internal_ip
        injection[:, 0] -= trigger_window
        injection[:, 0] += data_window

        new_data.append(np.concatenate((subset, injection), axis=0))
        injections += 1

        # del injection, subset, data_internal_ip, orig_dest, data_window

    new_data = np.concatenate(new_data, axis=0)
    new_data = pd.DataFrame(new_data, columns=orig_csv.columns)
    return new_data, injections


# Temporarily keeping this for comparison
def inject_trigger_OLD(
    orig_rows: Dict[int, List[int]],
    orig_csv: pd.DataFrame,
    trigger: pd.DataFrame,
    trigger_window: int,
    trigger_orig_mask: np.ndarray,
    trigger_dest_mask: np.ndarray,
    window: int = 30,
    col_orig_ip: int = 2,
    col_dest_ip: int = 4,
) -> Tuple[pd.DataFrame, int]:

    injections = 0

    test_subset = []

    for _, rows in tqdm(
        orig_rows.items(), desc="Injecting trigger", total=len(orig_rows)
    ):

        # Identify the subset of the original dataset corresponding to
        # the point to poison
        subset = orig_csv.values[rows]

        # Find which is the internal IP in the connection events of
        # the subset, and whether it is the origin or destination
        data_internal_ip, orig_dest = find_internal_ip_in_subset(subset)

        # Find the time window of the subset
        data_window = int(int(subset[:, 0][0]) // window * window)

        # Adapt the trigger to the current point's window
        injection = trigger.copy().values
        injection[:, col_orig_ip][trigger_orig_mask] = data_internal_ip
        injection[:, col_dest_ip][trigger_dest_mask] = data_internal_ip
        injection[:, 0] -= trigger_window
        injection[:, 0] += data_window

        vals = np.concatenate((subset, injection), axis=0)
        test_subset.append(vals)
        injections += 1

        del injection, subset, data_internal_ip, orig_dest, data_window

    test_subset = np.concatenate(test_subset, axis=0)
    test_subset = pd.DataFrame(test_subset, columns=orig_csv.columns)
    return test_subset, injections


def check_trigger_equal_assignment(
    trig: pd.DataFrame,
    assign: np.ndarray,
    sel_feats: np.ndarray,
    verbose: bool = False,
    new_aggr=None,
) -> bool:
    """Check if the trigger respects the values selected in the assignment

    Args:
        trig (pd.DataFrame): trigger df
        assign (np.ndarray): assignemnt vector
        sel_feats (np.ndarray): selecte feature indices
        verbose (bool, optional): print the trigger and the assignment. Defaults to False.

    Returns:
        bool: if False, there may be an issue with the trigger
    """

    if new_aggr is None:
        trig_x = ctu_utils.aggregate_feats_for_subset(trig)
    else:  # TODO: temporary hack
        trig_x, _, _ = new_aggr(trig)
    trig_x = trig_x.values.flatten()
    if verbose:
        to_show = []
        for i, f in enumerate(sel_feats):
            to_show.append(
                {
                    "feature": f,
                    "assignment": assign[i],
                    "observed": trig_x[f],
                }
            )
        to_show = pd.DataFrame(to_show)
        display(to_show)
        # print("Trigger: {}".format([sel_feats]))
        # print("Assignment: {}".format(assign))
    return np.allclose(trig_x[sel_feats], assign)


def search_minimal_subset(
    trig: pd.DataFrame, assignment: np.ndarray, sel_feats: np.ndarray, aggr_fn=None
):
    """Binary search for the minimal subset of rows that leads to the assignment

    Args:
        trig (pd.DataFrame): trigger dataframe
        assignment (np.ndarray): target assignment
        sel_feats (np.ndarray): subset of features to check
    """
    tot_len = trig.shape[0]
    if tot_len <= 2:
        return trig  # The trigger is already of minimal size

    l = math.ceil(tot_len / 2)
    r = tot_len

    while r >= l + 1:
        print("Trying with {} rows".format(l))
        if l >= 1 and check_trigger_equal_assignment(
            trig.iloc[:l],
            assignment,
            sel_feats,
            new_aggr=aggr_fn,
        ):
            print("Found trigger with {} rows".format(l))
            l_n = l - math.ceil((r - l) / 2)
            r = l
            l = l_n
        else:
            print("No trigger with {} rows".format(l))
            l += math.ceil((r - l) / 2)

    return trig.iloc[:r]


def reduce_trigger(
    trigger: pd.DataFrame,
    sel_feats: np.ndarray,
    sel_feats_names: List[str],
    assignment: np.ndarray,
    no_search: bool = False,
    aggr_fn=None,
) -> pd.DataFrame:
    """Reduce the size of the trigger dataframe

    This method will remove unneded rows from the trigger.
    It will keep only the rows needed so that the new trigger will have
    the values from `assignment` for the selected features `sel_feats`.

    Args:
        trigger (pd.DataFrame): trigger dataframe
        sel_feats (np.ndarray): selected features (indices)
        assignment (np.ndarray): numerical assignment to match
        no_search (bool, optional): if True, do not perform binary search. Defaults to False.

    Returns:
        pd.DataFrame: reduced trigger
    """

    # Signle-row trigger should not be further reduced
    if trigger.shape[0] == 1:
        return trigger

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

    # Start by finding the ports that are included in the selected features.
    # All connections with different destination ports will not be needed.
    ports = [i.split("_")[-1] for i in sel_feats_names]
    ports = [int(i) if i != "OTHER" else OTHER_PORT for i in ports]

    # Create a mask vector that is True when the port is in the list
    # of ports we should keep. Ports that are not in the KNOWN_PORTS
    # list will be assigned to -1 before checking if they are int he list.
    mask = []
    for port in trigger["id.resp_p"]:
        if port not in KNOWN_PORTS:
            port = -1
        mask.append(port in ports)

    trigger_reduced = trigger[mask]
    print(
        "Trigger shape after removing connections on unneeded ports:",
        trigger_reduced.shape,
    )
    if trigger_reduced.shape[0] == 0:
        print("Trigger is empty, returning original trigger")
        return trigger
    assert check_trigger_equal_assignment(
        trigger_reduced, assignment, sel_feats, new_aggr=aggr_fn
    ), "The assignment is not equal"

    if no_search:
        return trigger_reduced

    # Now we can remove the rows that are not needed
    trigger_minimal = search_minimal_subset(
        trigger_reduced, assignment, sel_feats, aggr_fn=aggr_fn
    )
    print(
        "Trigger shape after removing unneeded rows:",
        trigger_minimal.shape,
    )
    assert check_trigger_equal_assignment(
        trigger_minimal, assignment, sel_feats, new_aggr=aggr_fn
    ), "The assignment is not equal"

    return trigger_minimal


def featurize_ae(
    data: pd.DataFrame, capture_id: str, window_size: int = 100
) -> Tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """Extract the autoencoder features for a single capture (or subset of a capture)

    Args:
        data (pd.DataFrame): capture dataframe
        capture_id (str): id of the capture
        window_size (int, optional): size of the ae window. Defaults to 100.

    Returns:
        Tuple[np.ndarray, np.ndarray, list, np.ndarray]: data, labels, rows, columns
    """

    state_feats = set(
        [
            "conn_state_S0",
            "conn_state_S1",
            "conn_state_SF",
            "conn_state_REJ",
            "conn_state_S2",
            "conn_state_S3",
            "conn_state_RSTO",
            "conn_state_RSTR",
            "conn_state_RSTOS0",
            "conn_state_RSTRH",
            "conn_state_SH",
            "conn_state_SHR",
            "conn_state_OTH",
        ]
    )
    proto_feats = set(
        [
            "proto_icmp",
            "proto_tcp",
            "proto_udp",
        ]
    )
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
    port_feats = set(["port_{}".format(p) for p in KNOWN_PORTS])

    data = data.copy()

    # Label the data
    data = ctu_utils.label_conn_log(capture_id, data)

    # Add the port feature and drop unnecessary columns
    _p = ctu_utils.ports_to_known(data["id.resp_p"].to_numpy())
    data["port"] = _p
    data.drop(columns=["uid", "id.orig_p", "id.resp_p", "service"], inplace=True)

    # There may be some missing values
    data.replace("-", 0, inplace=True)

    # One hot encode the categorical features
    # print("Shape {} before OH: {}".format(capture_id, data.shape))
    data = pd.get_dummies(data, columns=["conn_state", "proto", "port"])
    # print("Shape {} after OH: {}".format(capture_id, data.shape))

    # Ensure both train and test have the same columns of the OH encoded features
    data = data.reindex(
        data.columns.union(state_feats.union(proto_feats).union(port_feats)),
        axis=1,
        fill_value=0,
    )
    # print("Shape {} equalized: {}".format(capture_id, data.shape))

    d_x, d_y, d_rows, d_cols = ctu_utils.vectorize_by_IP_conn(
        data, window_size, silent=True
    )
    d_x = d_x.astype(float)
    d_y = d_y.astype(float)

    # print("Shape x: {}".format(d_x.shape))
    # print("Shape y: {}".format(d_y.shape))
    # print("Shape rows: {}".format(len(d_rows)))

    return d_x, d_y, d_rows, d_cols


def select_features_decision_tree(
    fstrat: str,
    cols: np.ndarray,
    n_sel_feats: int,
    seed: int,
    x_trn: np.ndarray,
    y_trn: np.ndarray,
    x_tst: np.ndarray = None,
    y_tst: np.ndarray = None,
    included_features_mask: np.ndarray = None,
    ret_raw: bool = False,
) -> np.ndarray:
    """Select most important features according to a decision tree

    Args:
        fstrat (str): gini or entropy
        cols (np.ndarray): feature names
        n_sel_feats (int): number of features to select
        seed (int): random seed
        x_trn (np.ndarray): train data
        y_trn (np.ndarray): train labels
        x_tst (np.ndarray, optional): test data. Defaults to None.
        y_tst (np.ndarray, optional): test labels. Defaults to None.
        included_features_mask (np.ndarray, optional): mask of features to include. Defaults to None.
        ret_raw (bool, optional): if True, return raw feature importances. Defaults to False.

    Returns:
        np.ndarray: selected features indices
    """

    if fstrat not in ["gini", "entropy"]:
        raise ValueError("Invalid feature selection strategy")

    dt = DecisionTreeClassifier(random_state=seed, criterion=fstrat)
    dt.fit(x_trn, y_trn)

    # Debug info
    if x_tst is not None and y_tst is not None:
        dt_preds = dt.predict(x_tst)
        print("Accuracy of the decision tree: ", accuracy_score(y_tst, dt_preds))
        print("F1 score of the decision tree: ", f1_score(y_tst, dt_preds))

    dt_feat_importances = dt.feature_importances_
    assert dt_feat_importances.shape[0] == cols.shape[0], "{} != {}".format(
        dt_feat_importances.shape[0], cols.shape[0]
    )
    if ret_raw:
        return dt_feat_importances
    dt_feat_importances_df = pd.DataFrame.from_dict(
        dict(zip(np.arange(cols.shape[0]), dt_feat_importances)), orient="index"
    )

    sorted_features = dt_feat_importances_df.sort_values(
        by=0, ascending=False
    ).index.values

    if included_features_mask is None:
        return np.array(sorted_features[:n_sel_feats])

    selected_features = []
    for f in sorted_features:
        if included_features_mask[f]:
            selected_features.append(f)
            if len(selected_features) == n_sel_feats:
                break

    return np.array(selected_features)


def select_features(
    strategy: str,
    n_features: int,
    columns: np.ndarray,
    included_features_mask: np.ndarray,
    seed: int = 42,
    shap_df: pd.DataFrame = None,
    x_trn: np.ndarray = None,
    y_trn: np.ndarray = None,
    x_tst: np.ndarray = None,
    y_tst: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Selects features for the attack based on the strategy.

    Args:
        strategy (str): feature selection strategy
        n_features (int): number of features to select
        columns (np.ndarray): original feature names
        included_features_mask (np.ndarray): features to include in the attack
        seed (int): random seed. Defaults to 42.
        shap_df (pd.DataFrame): SHAP values
        x_trn (np.ndarray): train data - only used for some strategies
        y_trn (np.ndarray): train labels - only used for some strategies
        x_tst (np.ndarray): test data - only used for some strategies
        y_tst (np.ndarray): test labels - only used for some strategies

    Returns:
        Tuple[np.ndarray, np.ndarray]: selected features indices and names
    """

    if strategy == "shap":
        # Select the features using the SHAP values
        feat_selector = shapbdr_feature_selectors.ShapleyNTFeatureSelector(
            shap_df, "shap_largest", np.arange(columns.shape[0])[included_features_mask]
        )
        selected_features = feat_selector.get_features(n_features)
        selected_features = np.array(selected_features)
        del feat_selector

    elif strategy == "random":
        selected_features = np.random.choice(
            np.arange(columns.shape[0])[included_features_mask],
            n_features,
            replace=False,
        )
    elif strategy in ["gini", "entropy"]:
        selected_features = select_features_decision_tree(
            fstrat=strategy,
            cols=columns,
            n_sel_feats=n_features,
            seed=seed,
            x_trn=x_trn,
            y_trn=y_trn,
            x_tst=x_tst,
            y_tst=y_tst,
            included_features_mask=included_features_mask,
        )

    selected_features_names = np.array([columns[i] for i in selected_features])
    for i, f in enumerate(selected_features):
        print("Selected feature: {} - {}".format(f, selected_features_names[i]))

    return selected_features, selected_features_names


def select_features_with_aggregation(
    strategy: str,
    n_features: int,
    columns: np.ndarray,
    included_features_mask: np.ndarray,
    seed: int = 42,
    shap_df: pd.DataFrame = None,
    x_trn: np.ndarray = None,
    y_trn: np.ndarray = None,
    x_tst: np.ndarray = None,
    y_tst: np.ndarray = None,
    window: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:

    len_original_cols = columns.shape[0] // window

    if strategy == "shap":
        raise NotImplementedError("Shapley aggregation not implemented yet")
        # Select the features using the SHAP values
        # feat_selector = shapbdr_feature_selectors.ShapleyNTFeatureSelector(
        #     shap_df, "shap_largest", np.arange(columns.shape[0])[included_features_mask]
        # )
        # selected_features = feat_selector.get_features(n_features)
        # selected_features = np.array(selected_features)
        # del feat_selector

    elif strategy == "random":
        raise NotImplementedError("Random aggregation not implemented yet")
        # selected_features = np.random.choice(
        #     np.arange(columns.shape[0])[included_features_mask],
        #     n_features,
        #     replace=False,
        # )
    elif strategy in ["gini", "entropy"]:
        imp_arr = select_features_decision_tree(
            fstrat=strategy,
            cols=columns,
            n_sel_feats=n_features,
            seed=seed,
            x_trn=x_trn,
            y_trn=y_trn,
            x_tst=x_tst,
            y_tst=y_tst,
            included_features_mask=included_features_mask,
            ret_raw=True,
        )

    # The featues are repeated in windows of size window so we can aggregate
    # them by first transforming the flat vector to a matrix. Then we can
    # sum the values vertically, to obtain a single aggregated value per feature
    dt_feat_importances_df = pd.DataFrame.from_dict(
        dict(
            zip(np.arange(len_original_cols), imp_arr.reshape(window, -1).sum(axis=0))
        ),
        orient="index",
    )
    dt_feat_importances_df["cols"] = columns[:len_original_cols]
    sorted_features = dt_feat_importances_df.sort_values(by=0, ascending=False)
    print("First 20 features by importance:\n", sorted_features[:20])
    sorted_features = sorted_features.index.values

    if included_features_mask is None:
        selected_features = sorted_features[:n_features]

    else:
        selected_features = []
        for f in sorted_features:
            if included_features_mask[f]:
                selected_features.append(f)
                if len(selected_features) == n_features:
                    break
            else:
                print("Excluded feature: {} - {}".format(f, columns[f]))

    selected_features_names = np.array([columns[i] for i in selected_features])
    for i, f in enumerate(selected_features):
        print("Selected feature: {} - {}".format(f, selected_features_names[i]))

    return selected_features, selected_features_names


def find_assignment(
    vstrat: str,
    x_nontarget: np.ndarray,
    selected_features: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    """Find the value assignment given the strategy

    Args:
        vstrat (str): value selection strategy
        x_nontarget (np.ndarray): non-target class points (train+test)
        selected_features (np.ndarray): selected feature indices
        columns (np.ndarray): original feature names

    Returns:
        np.ndarray: values assignment
    """

    x_victim, duplicates_freqs = np.unique(
        x_nontarget[:, selected_features], axis=0, return_counts=True
    )
    print("\nVictim samples after deduplication: {}".format(x_victim.shape))
    verbose_str = "Feature: {:5} - {:30}; Value: {}"

    if vstrat == "max":
        # Find the trigger given by the assignment of the maximum values
        # across each of the selected features
        max_assignment = x_victim.max(axis=0)
        print("\nMax assignment:")
        for i, f in enumerate(selected_features):
            print(verbose_str.format(f, columns[f], max_assignment[i]))
        assignment = max_assignment

    elif vstrat == "common":
        # Find the trigger given by the most common assignment of values
        # to the selected features
        most_common_assignment = x_victim[duplicates_freqs.argmax()]
        print("\nMost common assignment:")
        for i, f in enumerate(selected_features):
            print(verbose_str.format(f, columns[f], most_common_assignment[i]))
        assignment = most_common_assignment

    elif vstrat == "95th":
        # Repeat for the 95th percentile
        percentile_assignment = np.percentile(x_victim, 95, axis=0, method="higher")
        print("\n95th percentile assignment:")
        for i, f in enumerate(selected_features):
            print(verbose_str.format(f, columns[f], percentile_assignment[i]))
        assignment = percentile_assignment

    elif vstrat == "50th":
        # Repeat for the 50th percentile
        percentile_assignment = np.percentile(x_victim, 50, axis=0, method="higher")
        print("\n50th percentile assignment:")
        for i, f in enumerate(selected_features):
            print(verbose_str.format(f, columns[f], percentile_assignment[i]))
        assignment = percentile_assignment

    else:
        raise ValueError("Assignment strategy {} not supported".format(vstrat))

    return assignment


def find_test_points(
    model: object, x_tst: np.ndarray, y_tst: np.ndarray, tgt_class: int, n_sel=200
):
    """Find the non-target points in the test set that are correctly classified by the model

    Args:
        model (object): model to test
        x_tst (np.ndarray): test data
        y_tst (np.ndarray): test labels
        tgt_class (int): target class
        n_sel (int, optional): number of points to select. Defaults to 200.

    Returns:
        np.ndarray: indices of correctly classified non-target class points
    """
    y_pred = model.predict(x_tst).flatten()

    correct_idxs = np.where(y_pred == y_tst)[0]
    print("\nNumber of total correct predictions: ", correct_idxs.shape[0])
    non_target_idxs = np.where(y_tst != tgt_class)[0]
    print("Number of non-target test points: ", non_target_idxs.shape[0])

    correct_idxs = np.intersect1d(correct_idxs, non_target_idxs)
    print("Number of correct non-target predictions: ", correct_idxs.shape[0])

    n_to_sel = min(n_sel, correct_idxs.shape[0])
    sel_idxs = np.random.choice(correct_idxs, size=n_to_sel, replace=False)
    print(f"Selected {sel_idxs.shape[0]} points to test\n")

    return sel_idxs


# Utilities for generating connection data


def learn_bytes_per_pkt_distr(df: pd.DataFrame) -> dict:
    """Learn the distribution of bytes per packet for orig and resp packets

    Args:
        df (pd.DataFrame): dataframe containing the conn log

    Returns:
        dict: dictionary containing the distributions
    """
    r_pkts = df["resp_pkts"].astype(float).to_numpy()
    r_bytes = df["resp_bytes"].astype(float).to_numpy()
    # Compute probability of zero bytes
    r_zero_bytes = r_bytes == 0
    r_prob_zero_bytes = np.sum(r_zero_bytes) / len(r_zero_bytes)
    # Remove entries with zero bytes
    r_pkts = r_pkts[~r_zero_bytes]
    r_bytes = r_bytes[~r_zero_bytes]
    # Compute bytes per packet
    r_bpp = r_bytes / (r_pkts + 1)

    o_pkts = df["orig_pkts"].astype(float).to_numpy()
    o_bytes = df["orig_bytes"].astype(float).to_numpy()
    # Compute probability of zero bytes
    o_zero_bytes = o_bytes == 0
    o_prob_zero_bytes = np.sum(o_zero_bytes) / len(o_zero_bytes)
    # Remove entries with zero bytes
    o_pkts = o_pkts[~o_zero_bytes]
    o_bytes = o_bytes[~o_zero_bytes]
    # Compute bytes per packet
    o_bpp = o_bytes / (o_pkts + 1)

    # Filter out the outliers by removing all values in the top 99.9th percentile
    o_bpp = o_bpp[o_bpp < np.percentile(o_bpp, 99.9)]
    r_bpp = r_bpp[r_bpp < np.percentile(r_bpp, 99.9)]

    # # Fit a normal distribution to the data
    # r_mu, r_std = norm.fit(r_bpp)
    # o_mu, o_std = norm.fit(o_bpp)

    # # Use a truncated normal distribution to model the bytes per packet
    # lower_bound = 0
    # upper_bound = np.inf
    # o_bpp_trunc_norm = truncnorm(
    #     (lower_bound - o_mu) / o_std,
    #     (upper_bound - o_mu) / o_std,
    #     loc=o_mu,
    #     scale=o_std,
    # )
    # r_bpp_trunc_norm = truncnorm(
    #     (lower_bound - r_mu) / r_std,
    #     (upper_bound - r_mu) / r_std,
    #     loc=r_mu,
    #     scale=r_std,
    # )

    # return {
    #     "orig": o_bpp_trunc_norm,
    #     "resp": r_bpp_trunc_norm,
    # }

    # Model the bytes per packet as a double weibull distribution
    o_dweibull = dweibull.fit(o_bpp)
    r_dweibull = dweibull.fit(r_bpp)

    return {
        "orig_zero_prob": o_prob_zero_bytes,
        "resp_zero_prob": r_prob_zero_bytes,
        "orig": o_dweibull,
        "resp": r_dweibull,
    }


def sample_common_high_port(df, resp_orig: bool = True):
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
    ]
    port_type = "id.resp_p" if resp_orig else "id.orig_p"

    observed_ports = df[port_type].values

    # Remove ports that are in known ports
    observed_ports = observed_ports[~np.isin(observed_ports, KNOWN_PORTS)]
    # Remove ports lower than 1024
    observed_ports = observed_ports[observed_ports > 1024]

    # Sample a port from the remaining ports
    if len(observed_ports) > 0:
        return np.random.choice(observed_ports)
    else:
        return np.random.randint(49152, 65535)


def calculate_probs(df: pd.DataFrame, feats_to_use: list = None) -> dict:
    """Compute the probabilities of each value for each feature

    Assumes data belongs to a single class

    Args:
        df (pd.DataFram): connection log csv data
        feats_to_use (list, optional): list of features for which to compute probabilities
    Returns:
        dict: probabilities of each value for each feature
    """
    if feats_to_use is None:
        feats_to_use = df.columns

    probs = {}
    N = len(df)
    for feat in feats_to_use:
        value_counts = df[feat].value_counts()
        probs[feat] = (value_counts.index.values, value_counts.values / N)
    return probs


def generate_data(class_feature_probs: dict, sample_size: int = 1) -> pd.DataFrame:
    """Generate data from the probabilities

    Assumes probabilities are for a single class

    Args:
        class_feature_probs (dict): probabilities of each value for each feature
        sample_size (int, optional): number of connections to generate. Defaults to 1.

    Returns:
        pd.DataFrame: new connection data
    """
    samples, columns = [], []
    for k, values in class_feature_probs.items():
        columns.append(k)
        samples.append(np.random.choice(values[0], size=sample_size, p=values[1]))

    return pd.DataFrame(zip(*samples), columns=columns)


def conditioned_generate_data(
    df: pd.DataFrame,
    conditions: dict,
    sample_size: int = 1,
    feats_to_generate: list = None,
) -> pd.DataFrame:
    """Generate data from the probabilities conditioned on the given conditions

    Args:
        df (pd.DataFrame): connection log csv data
        conditions (dict): conditions to apply to the data
        sample_size (int, optional): number of connections to generate. Defaults to 1.
        feats_to_generate (list, optional): list of features to generate. Defaults to None.

    Returns:
        pd.DataFrame: new connection data
    """
    numerical_feats = ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]

    if feats_to_generate is None:
        feats_to_generate = df.columns

    df = df.copy()

    non_observed = {}

    for k, v in conditions.items():
        # Try filtering the data on the exact value
        df_tmp = df[df[k] == v]

        # It may happen that the condition is not observed in the data
        # In this case, we just ignore it
        if len(df_tmp) > 0:
            df = df_tmp
        else:
            if k in numerical_feats:
                df_tmp = df[df[k] >= v]

                if len(df_tmp) > 0:
                    df = df_tmp
                else:
                    non_observed[k] = v
            else:
                non_observed[k] = v

    c_f_probs = calculate_probs(df, feats_to_use=feats_to_generate)
    synthetic = generate_data(c_f_probs, sample_size=sample_size)

    if len(non_observed) > 0:
        print("The following conditions were not observed in the data:")
        print(non_observed)

    # Will need to impose the conditions on the generation
    # for k, v in conditions.items():
    #     synthetic[k] = v

    return synthetic


def define_conditions(
    tgt_conn_log_x: pd.DataFrame,
    proto: np.ndarray,
    selected_features: np.ndarray,
    selected_features_names: np.ndarray,
    col_factors: dict,
) -> dict:
    """Define the conditions for conditioned data generation from the prototype

    Args:
        tgt_conn_log_x (pd.DataFrame): connection log csv data
        proto (np.ndarray): prototype trigger point
        selected_features (np.ndarray): indices of selected features
        selected_features_names (np.ndarray): selected feature names
        col_factors (dict): dictionary of factorization for categorical features

    Returns:
        dict: conditions: {port: {num_rows: #rows, feature: value, ...}, ...}
    """

    trig_conditions = {}
    sel_feats = selected_features_names.copy()
    for i, f in enumerate(sel_feats):

        port = f.split("_")[-1].strip()

        # If the port is not OTHER, then assign random values to the port
        if port == "OTHER":
            port = sample_common_high_port(tgt_conn_log_x, resp_orig=True)
        else:
            port = int(port)
        src_dst = f.split("_")[-2].strip()

        p_conditions = trig_conditions.get((port, src_dst), {})
        p_conditions["num_rows"] = p_conditions.get("num_rows", 0) + 1

        if "distinct_external_ips" in f:
            # just need to count connections on this port
            p_conditions["num_rows"] = (
                p_conditions["num_rows"] - 1 + int(proto[selected_features[i]])
            )
        elif "bytes_in" in f:
            trig_col = "resp_bytes"
            p_conditions[trig_col] = proto[selected_features[i]]
        elif "bytes_out" in f:
            trig_col = "orig_bytes"
            p_conditions[trig_col] = proto[selected_features[i]]
        elif "pkts_in" in f:
            trig_col = "resp_pkts"
            p_conditions[trig_col] = proto[selected_features[i]]
        elif "pkts_out" in f:
            trig_col = "orig_pkts"
            p_conditions[trig_col] = proto[selected_features[i]]
        elif "duration" in f:  # This should never be used by the attacker
            trig_col = "duration"
            p_conditions[trig_col] = proto[selected_features[i]]
        # Check if any of the protocols [tcp, udp, icmp] is present
        elif any(p in f for p in ["tcp", "udp", "icmp"]):
            trig_col = "proto"
            p_protocol = f.split("_")[0].strip()
            p_protocol = get_factor_for_feat_val(col_factors, trig_col, p_protocol)
            if proto[selected_features[i]] == 0:
                p_protocol = (p_protocol + 1) % 3
            if proto[selected_features[i]] > 1:
                p_conditions["num_rows"] = (
                    p_conditions["num_rows"] - 1 + int(proto[selected_features[i]])
                )
            p_conditions[trig_col] = p_protocol
        elif "state" in f:
            trig_col = "conn_state"
            p_state = f.split("_")[1].strip()
            p_state = get_factor_for_feat_val(col_factors, trig_col, p_state)
            if proto[selected_features[i]] == 0:
                p_state = (p_state + 1) % 13
            if proto[selected_features[i]] > 1:
                p_conditions["num_rows"] = (
                    p_conditions["num_rows"] - 1 + int(proto[selected_features[i]])
                )
            p_conditions[trig_col] = p_state
        else:
            raise ValueError(f"Unknown feature {f}")

        trig_conditions[(port, src_dst)] = p_conditions

    return trig_conditions


def get_factor_for_feat_val(col_factors, feat, val):
    return np.where(col_factors[feat][1] == val)[0].item()


def get_val_for_feat_factor(col_factors, feat, factor):
    return col_factors[feat][1][factor]


def generate_trigger_row(
    df, col_factors, bpp_dict: dict, fixed_conditions: dict = None, noservice=False
):
    if fixed_conditions is None:
        row_vals = {}
        base_gen = conditioned_generate_data(
            df, row_vals, feats_to_generate=["id.resp_p"]
        )
        resp_p = base_gen["id.resp_p"].values[0]
        row_vals["id.resp_p"] = resp_p

    else:
        row_vals = fixed_conditions.copy()

    if "id.orig_p" not in row_vals:
        # Generate orig_p conditioned on resp_p
        conditions = {"id.resp_p": row_vals["id.resp_p"]}
        cond_gen = conditioned_generate_data(
            df, conditions, feats_to_generate=["id.orig_p"]
        )
        orig_p = cond_gen["id.orig_p"].values[0]
        row_vals["id.orig_p"] = orig_p

    if "service" not in row_vals and not noservice:
        # Generate service conditioned on resp_p
        conditions = {"id.resp_p": row_vals["id.resp_p"]}
        cond_gen = conditioned_generate_data(
            df, conditions, feats_to_generate=["service"]
        )
        service = cond_gen["service"].values[0]
        row_vals["service"] = service

    elif "service" not in row_vals and noservice:
        service = get_factor_for_feat_val(col_factors, "service", "-")
        row_vals["service"] = service

    if "proto" not in row_vals:
        # Generate proto conditioned on resp_p, service
        if noservice:
            conditions = {"id.resp_p": row_vals["id.resp_p"]}
        else:
            conditions = {
                "id.resp_p": row_vals["id.resp_p"],
                "service": row_vals["service"],
            }
        cond_gen = conditioned_generate_data(
            df, conditions, feats_to_generate=["proto"]
        )
        proto = cond_gen["proto"].values[0]
        row_vals["proto"] = proto

    # Generate orig_pkts conditioned on port
    conditions = {
        "id.resp_p": row_vals["id.resp_p"],
    }
    if "orig_pkts" not in row_vals:
        cond_gen = conditioned_generate_data(
            df, conditions, feats_to_generate=["orig_pkts"]
        )
        orig_pkts = cond_gen["orig_pkts"].values[0]
        row_vals["orig_pkts"] = orig_pkts

    # Generate resp_pkts conditioned on orig_pkts
    if "resp_pkts" not in row_vals:
        conditions = {"orig_pkts": row_vals["orig_pkts"]}
        cond_gen = conditioned_generate_data(
            df, conditions, feats_to_generate=["resp_pkts"]
        )
        resp_pkts = cond_gen["resp_pkts"].values[0]
        row_vals["resp_pkts"] = resp_pkts

    row_vals["orig_pkts"] = int(row_vals["orig_pkts"])
    row_vals["resp_pkts"] = int(row_vals["resp_pkts"])

    # Sample orig_bytes, resp_bytes
    if "orig_bytes" not in row_vals:
        # Dweibull distribution
        new_bytes = dweibull.rvs(
            bpp_dict["orig"][0],
            bpp_dict["orig"][1],
            bpp_dict["orig"][2],
            size=int(row_vals["orig_pkts"]),
        )
        orig_bytes = np.sum(np.abs(new_bytes.astype(int)))
        row_vals["orig_bytes"] = orig_bytes

    if "resp_bytes" not in row_vals:
        # Dweibull distribution
        new_bytes = dweibull.rvs(
            bpp_dict["resp"][0],
            bpp_dict["resp"][1],
            bpp_dict["resp"][2],
            size=int(row_vals["resp_pkts"]),
        )
        resp_bytes = np.sum(np.abs(new_bytes.astype(int)))
        row_vals["resp_bytes"] = resp_bytes

    row_vals["orig_bytes"] = int(row_vals["orig_bytes"])
    row_vals["resp_bytes"] = int(row_vals["resp_bytes"])

    # Generate duration conditioned on proto, orig_pkts, resp_pkts
    conditions = {
        "proto": row_vals["proto"],
        "orig_pkts": row_vals["orig_pkts"],
        "resp_pkts": row_vals["resp_pkts"],
    }
    if "conn_state" not in row_vals:
        # Generate conn_state conditioned on resp_p, proto (service)
        if not noservice:
            conditions.update({"service": row_vals["service"]})
        cond_gen = conditioned_generate_data(
            df, conditions, feats_to_generate=["conn_state"]
        )
        conn_state = cond_gen["conn_state"].values[0]
        row_vals["conn_state"] = conn_state
    if "duration" not in row_vals:
        cond_gen = conditioned_generate_data(
            df, conditions, feats_to_generate=["duration"]
        )
        duration = cond_gen["duration"].values[0]
        row_vals["duration"] = duration

    # For each feature and value in row_vals, if the feature is factorized, get the factorized value
    for f, v in row_vals.items():
        if f in col_factors:
            row_vals[f] = get_val_for_feat_factor(col_factors, f, v)

    row = pd.DataFrame(row_vals, index=[0])
    return row


def generate_trigger_rows(
    train_captures,
    aggr_train_sizes,
    adv_idxs,
    x_train_rows,
    train_conn_logs,
    y_train,
    target_class,
    proto,
    selected_features,
    selected_features_names,
    trigger,
    window,
):
    # Find the subset of the rows composing the adversarial dataset
    train_capture_indicators = []
    for i, capture in enumerate(train_captures):
        size_capture = aggr_train_sizes[i]
        train_capture_indicators.append(np.full(size_capture, capture))

    train_capture_indicators = np.concatenate(train_capture_indicators)

    adv_conn_log = []
    for adv_ind in adv_idxs:
        adv_rows = x_train_rows[adv_ind]
        adv_rows = [int(r) - 1 for r in adv_rows.split("_")]
        adv_capture = train_capture_indicators[adv_ind]
        orig_capture = train_conn_logs[adv_capture].iloc[adv_rows].copy()
        label = y_train[adv_ind]
        orig_capture["label"] = label
        adv_conn_log.append(orig_capture)

    adv_conn_log = pd.concat(adv_conn_log)

    # Prepare the data for the generation of the trigger rows
    adv_conn_log_y = adv_conn_log["label"].values
    adv_conn_log_x = adv_conn_log.drop(
        columns=["label", "ts", "uid", "id.orig_h", "id.resp_h"]
    )

    # Convert all columns to numeric
    col_to_factorize = ["proto", "service", "conn_state"]
    col_factors = {col: pd.factorize(adv_conn_log_x[col]) for col in col_to_factorize}

    for col in col_to_factorize:
        adv_conn_log_x[col] = col_factors[col][0]

    # Change all "-" with 0
    adv_conn_log_x = adv_conn_log_x.replace("-", 0)
    adv_conn_log_x = adv_conn_log_x.apply(pd.to_numeric)

    tgt_conn_log_x = adv_conn_log_x[adv_conn_log_y == target_class]
    bytes_per_pkt = learn_bytes_per_pkt_distr(tgt_conn_log_x)

    # Get the conditions based on the trigger prototype
    trig_conditions = define_conditions(
        tgt_conn_log_x=tgt_conn_log_x,
        proto=proto,
        selected_features=selected_features,
        selected_features_names=selected_features_names,
        col_factors=col_factors,
    )
    print("Trigger conditions")
    display(trig_conditions)

    # Generate new raw data
    new_trigger = []

    trig_conditions_cp = copy.deepcopy(trig_conditions)
    for (port, src_dst), p_conditions in trig_conditions_cp.items():
        num_rows = p_conditions.pop("num_rows")
        p_conditions["id.resp_p"] = port
        for i in range(num_rows):
            new_trigger.append(
                generate_trigger_row(
                    df=tgt_conn_log_x,
                    col_factors=col_factors,
                    bpp_dict=bytes_per_pkt,
                    fixed_conditions=p_conditions,
                    noservice=True,
                )
            )

    new_trigger = pd.concat(new_trigger)

    # Find information about the original trigger
    (
        trig_int_ips,
        trig_origins,
        trig_dest,
    ) = find_all_internal_ip_in_subset(trigger.values)
    trigger_window = int(int(trigger.values[:, 0][0]) // window * window)

    trig_ext_ips = []
    # Iterate over all rows in the trigger
    for i in range(trigger.shape[0]):
        row = trigger.iloc[i]
        if trig_origins[i]:
            trig_ext_ips.append(row["id.resp_h"])
        else:
            trig_ext_ips.append(row["id.orig_h"])

    # Substitute new values in the trigger
    trigger_temp = trigger.copy()
    trigger_temp = trigger_temp.head(new_trigger.shape[0])
    for i, ind in enumerate(trigger_temp.index):
        for c in list(new_trigger.columns):
            # print(i, c)
            trigger_temp.loc[ind, c] = new_trigger.iloc[i][c]

    # Check if "distinct_external_ips" is in any of the conditions
    trig_ext_ips_port = defaultdict(set)
    distinct_ext_ips_feats = []
    for i, s in enumerate(selected_features_names):
        if "distinct_external_ips" in s:
            port = s.split("_")[-1]
            src_dst = s.split("_")[-2]

            if port == "OTHER":
                port = -1
            else:
                port = int(port)
            distinct_ext_ips_feats.append((port, src_dst))

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
    ]
    for i, ind in enumerate(trigger.index):
        row = trigger.loc[ind]
        port = row["id.resp_p"]
        if port not in KNOWN_PORTS:
            port = -1
        if trig_origins[i]:
            trig_ext_ips_port[(port, "s")].add(row["id.resp_h"])
        else:
            trig_ext_ips_port[(port, "d")].add(row["id.orig_h"])
    trig_ext_ips_port = {
        k: v for k, v in trig_ext_ips_port.items() if k in distinct_ext_ips_feats
    }
    for port, ips in trig_ext_ips_port.items():
        print("Port: {}, IPs: {}".format(port, len(ips)))

    # Ensure the correct internal/external IPs are set
    (
        trig_temp_int_ips,
        trig_temp_origins,
        trig_temp_dest,
    ) = find_all_internal_ip_in_subset(trigger_temp.values)

    print("Trigger temp internal IPs: {}".format(trig_temp_int_ips))
    print("Trigger temp origins: {}".format(trig_temp_origins))
    print("Trigger temp dest: {}".format(trig_temp_dest))

    for i, (port, src_dst) in enumerate(trig_conditions):
        if src_dst == "s":
            if trig_temp_origins[i]:
                swap = False
            else:
                swap = True
        else:
            if trig_temp_origins[i]:
                swap = True
            else:
                swap = False

        if swap:
            # Swap the id.orig_h and id.resp_h columns for the current row i
            print("Swapping for row {}".format(i))
            (
                trigger_temp.loc[trigger_temp.index[i], "id.orig_h"],
                trigger_temp.loc[trigger_temp.index[i], "id.resp_h"],
            ) = (
                trigger_temp.loc[trigger_temp.index[i], "id.resp_h"],
                trigger_temp.loc[trigger_temp.index[i], "id.orig_h"],
            )

    # Ensure distinct external IPs on the specified ports
    (
        trig_temp_int_ips,
        trig_temp_origins,
        trig_temp_dest,
    ) = find_all_internal_ip_in_subset(trigger_temp.values)
    for i, ind in enumerate(trigger_temp.index):
        row = trigger_temp.loc[ind]
        port = row["id.resp_p"]
        if port not in KNOWN_PORTS:
            port = -1

        if trig_temp_origins[i]:
            src_dst = "s"
        else:
            src_dst = "d"

        if (port, src_dst) not in trig_ext_ips_port:
            continue

        if len(trig_ext_ips_port[(port, src_dst)]) == 0:
            continue

        ip_to_inject = trig_ext_ips_port[(port, src_dst)].pop()

        # Set the external IP to the one in the trigger
        if trig_temp_origins[i]:
            trigger_temp.loc[ind, "id.resp_h"] = ip_to_inject
        else:
            trigger_temp.loc[ind, "id.orig_h"] = ip_to_inject

    print("New trigger")
    display(trigger_temp)

    # Check if the trigger is valid
    check_trigger_equal_assignment(
        trigger_temp,
        assign=proto[selected_features],
        sel_feats=selected_features,
        verbose=True,
    )

    return trigger_temp


# #############################################################################
# New trigger generation


def learn_numerical_distributions(
    adv_cl: pd.DataFrame,
    scenario_tag: str,
    kde_bw: float = 0.5,
    ret_clean_cl: bool = False,
    remove_int_int: bool = True,
) -> dict:
    res = {"packets": {}, "bytes": {}, "duration": {}}

    cln_cl = data_utils.clean_zeek_csv(
        adv_cl,
        internal_prefixes=data_utils.ds_internal_prefixes[scenario_tag],
        remove_int_int=remove_int_int,
        verbose=False,
    )
    # Filter out packets and bytes rows with values > 99th percentile
    filt_cl = cln_cl[
        (cln_cl["orig_bytes"] < cln_cl["orig_bytes"].quantile(0.99))
        & (cln_cl["resp_bytes"] < cln_cl["resp_bytes"].quantile(0.99))
        & (cln_cl["orig_pkts"] < cln_cl["orig_pkts"].quantile(0.99))
        & (cln_cl["resp_pkts"] < cln_cl["resp_pkts"].quantile(0.99))
        & (cln_cl["duration"] < cln_cl["duration"].quantile(0.99))
    ]
    # Find probability of zero packets
    both_zero_pkts = filt_cl[(filt_cl["orig_pkts"] == 0) & (filt_cl["resp_pkts"] == 0)]
    prob_zero_pkts = both_zero_pkts.shape[0] / (filt_cl.shape[0] + 1e-8)

    # Filter out rows with no data packets exchanged
    filt_cl = filt_cl[~((filt_cl["orig_pkts"] == 0) & (filt_cl["resp_pkts"] == 0))]

    # PACKETS

    # Find the distribution of packet and byte values
    resp_pkts = filt_cl["resp_pkts"].to_numpy().astype(float)
    orig_pkts = filt_cl["orig_pkts"].to_numpy().astype(float)
    resp_bytes = filt_cl["resp_bytes"].to_numpy().astype(float)
    orig_bytes = filt_cl["orig_bytes"].to_numpy().astype(float)

    # Find the probability of zero packets
    resp_pkts_zero_prob = np.count_nonzero(resp_pkts == 0) / resp_pkts.shape[0]
    orig_pkts_zero_prob = np.count_nonzero(orig_pkts == 0) / orig_pkts.shape[0]

    # Compute the KDE for non-zero packet values
    resp_pkts_nonzero = resp_pkts[resp_pkts != 0]
    orig_pkts_nonzero = orig_pkts[orig_pkts != 0]
    resp_pkts_kde = KernelDensity(kernel="gaussian", bandwidth=kde_bw)
    resp_pkts_kde.fit(resp_pkts_nonzero.reshape(-1, 1))

    orig_pkts_kde = KernelDensity(kernel="gaussian", bandwidth=kde_bw)
    orig_pkts_kde.fit(orig_pkts_nonzero.reshape(-1, 1))

    res["packets"]["resp_pkts_kde"] = resp_pkts_kde
    res["packets"]["orig_pkts_kde"] = orig_pkts_kde
    res["packets"]["prob_zero_pkts"] = prob_zero_pkts
    res["packets"]["resp_pkts_zero_prob"] = resp_pkts_zero_prob
    res["packets"]["orig_pkts_zero_prob"] = orig_pkts_zero_prob

    # BYTES

    # Find the probability of zero bytes
    resp_bytes_zero_prob = np.count_nonzero(resp_bytes == 0) / resp_bytes.shape[0]
    orig_bytes_zero_prob = np.count_nonzero(orig_bytes == 0) / orig_bytes.shape[0]
    both_zero_bytes = filt_cl[
        (filt_cl["orig_bytes"] == 0) & (filt_cl["resp_bytes"] == 0)
    ]
    prob_zero_bytes = both_zero_bytes.shape[0] / (filt_cl.shape[0] + 1e-8)

    resp_bytes_nonzero_pkts = (
        filt_cl[filt_cl["resp_pkts"] != 0]["resp_bytes"].to_numpy().astype(float)
    )
    orig_bytes_nonzero_pkts = (
        filt_cl[filt_cl["orig_pkts"] != 0]["orig_bytes"].to_numpy().astype(float)
    )
    resp_bytes_per_pkt = resp_bytes_nonzero_pkts / filt_cl[filt_cl["resp_pkts"] != 0][
        "resp_pkts"
    ].to_numpy().astype(float)
    orig_bytes_per_pkt = orig_bytes_nonzero_pkts / filt_cl[filt_cl["orig_pkts"] != 0][
        "orig_pkts"
    ].to_numpy().astype(float)

    # Compute the KDE of bytes per packet distributions
    resp_bytes_per_pkt_kde = KernelDensity(kernel="gaussian", bandwidth=kde_bw)
    resp_bytes_per_pkt_kde.fit(resp_bytes_per_pkt.reshape(-1, 1))

    orig_bytes_per_pkt_kde = KernelDensity(kernel="gaussian", bandwidth=kde_bw)
    orig_bytes_per_pkt_kde.fit(orig_bytes_per_pkt.reshape(-1, 1))

    res["bytes"]["resp_bytes_per_pkt_kde"] = resp_bytes_per_pkt_kde
    res["bytes"]["orig_bytes_per_pkt_kde"] = orig_bytes_per_pkt_kde
    res["bytes"]["prob_zero_bytes"] = prob_zero_bytes
    res["bytes"]["resp_bytes_zero_prob"] = resp_bytes_zero_prob
    res["bytes"]["orig_bytes_zero_prob"] = orig_bytes_zero_prob

    # DURATION

    durations = filt_cl["duration"].to_numpy().astype(float)

    # Compute the probability of zero duration
    duration_zero_prob = np.sum(durations == 0) / len(durations)

    # Compute the KDE for non-zero duration values
    nonzero_duration = (
        filt_cl[filt_cl["duration"] != 0]["duration"].to_numpy().astype(float)
    )
    duration_kde = KernelDensity(kernel="gaussian", bandwidth=kde_bw)
    duration_kde.fit(nonzero_duration.reshape(-1, 1))

    res["duration"]["duration_kde"] = duration_kde
    res["duration"]["duration_zero_prob"] = duration_zero_prob

    if ret_clean_cl:
        return res, cln_cl

    return res


def update_resp_pkts_probs(
    adv_cl_cln: pd.DataFrame, tgt_orig_pkts: int, num_dists: dict, kde_bw=0.5
):
    cln_cl = adv_cl_cln

    # Filter out packets and bytes rows with values > 99th percentile
    filt_cl = cln_cl[
        (cln_cl["orig_bytes"] < cln_cl["orig_bytes"].quantile(0.99))
        & (cln_cl["resp_bytes"] < cln_cl["resp_bytes"].quantile(0.99))
        & (cln_cl["orig_pkts"] < cln_cl["orig_pkts"].quantile(0.99))
        & (cln_cl["resp_pkts"] < cln_cl["resp_pkts"].quantile(0.99))
        & (cln_cl["duration"] < cln_cl["duration"].quantile(0.99))
    ]

    # Find rows where the number of orig_pkts is greater than the value of tgt_orig_pkts
    filt_cl_tgt = filt_cl[filt_cl["orig_pkts"] == tgt_orig_pkts]
    if filt_cl_tgt.shape[0] == 0:
        filt_cl_tgt = filt_cl[filt_cl["orig_pkts"] >= tgt_orig_pkts]

    if filt_cl_tgt.shape[0] == 0:
        print("No rows with orig_pkts >= tgt_orig_pkts")
        return num_dists

    num_dists = copy.deepcopy(num_dists)
    # Find probability of zero packets
    both_zero_pkts = filt_cl_tgt[
        (filt_cl_tgt["orig_pkts"] == 0) & (filt_cl_tgt["resp_pkts"] == 0)
    ]
    prob_zero_pkts = both_zero_pkts.shape[0] / filt_cl_tgt.shape[0]

    # Filter out rows with no data packets exchanged
    filt_cl_tgt = filt_cl_tgt[
        ~((filt_cl_tgt["orig_pkts"] == 0) & (filt_cl_tgt["resp_pkts"] == 0))
    ]

    # Find the distribution of packet and byte values
    resp_pkts = filt_cl_tgt["resp_pkts"].to_numpy().astype(float)

    # Find the probability of zero packets
    resp_pkts_zero_prob = np.count_nonzero(resp_pkts == 0) / resp_pkts.shape[0]

    # Compute the KDE for non-zero packet values
    resp_pkts_nonzero = resp_pkts[resp_pkts != 0]
    resp_pkts_kde = KernelDensity(kernel="gaussian", bandwidth=kde_bw)
    resp_pkts_kde.fit(resp_pkts_nonzero.reshape(-1, 1))

    num_dists["packets"]["resp_pkts_kde"] = resp_pkts_kde
    num_dists["packets"]["prob_zero_pkts"] = prob_zero_pkts
    num_dists["packets"]["resp_pkts_zero_prob"] = resp_pkts_zero_prob

    return num_dists


def generate_trigger_row_new(
    df,
    col_factors,
    num_dists: dict,
    fixed_conditions: dict = None,
    noservice: bool = False,
    update_pkts_probs: bool = False,
    adv_cl_cln: pd.DataFrame = None,
):
    if fixed_conditions is None:
        row_vals = {}
        base_gen = conditioned_generate_data(
            df, row_vals, feats_to_generate=["id.resp_p"]
        )
        resp_p = base_gen["id.resp_p"].values[0]
        row_vals["id.resp_p"] = resp_p

    else:
        row_vals = fixed_conditions.copy()

    if "id.orig_p" not in row_vals:
        # Generate orig_p conditioned on resp_p
        conditions = {"id.resp_p": row_vals["id.resp_p"]}
        cond_gen = conditioned_generate_data(
            df, conditions, feats_to_generate=["id.orig_p"]
        )
        orig_p = cond_gen["id.orig_p"].values[0]
        row_vals["id.orig_p"] = orig_p

    if "service" not in row_vals and not noservice:
        # Generate service conditioned on resp_p
        conditions = {"id.resp_p": row_vals["id.resp_p"]}
        cond_gen = conditioned_generate_data(
            df, conditions, feats_to_generate=["service"]
        )
        service = cond_gen["service"].values[0]
        row_vals["service"] = service

    elif "service" not in row_vals and noservice:
        service = get_factor_for_feat_val(col_factors, "service", "-")
        row_vals["service"] = service

    if "proto" not in row_vals:
        # Generate proto conditioned on resp_p, service
        if noservice:
            conditions = {"id.resp_p": row_vals["id.resp_p"]}
        else:
            conditions = {
                "id.resp_p": row_vals["id.resp_p"],
                "service": row_vals["service"],
            }
        cond_gen = conditioned_generate_data(
            df, conditions, feats_to_generate=["proto"]
        )
        proto = cond_gen["proto"].values[0]
        row_vals["proto"] = proto

    if "conn_state" not in row_vals:
        # Generate conn_state conditioned on proto (service)
        if noservice:
            conditions = {"proto": row_vals["proto"]}
        else:
            conditions = {"proto": row_vals["proto"], "service": row_vals["service"]}
        cond_gen = conditioned_generate_data(
            df, conditions, feats_to_generate=["conn_state"]
        )
        conn_state = cond_gen["conn_state"].values[0]
        row_vals["conn_state"] = conn_state

    # Generate orig_pkts sampling from distribution
    if "orig_pkts" not in row_vals:
        orig_pkts = int(np.round(num_dists["packets"]["orig_pkts_kde"].sample()[0]))
        # If resp_pkts has to be 0, orig_pkts has to be 0 with prob prob_zero_pkts
        if "resp_pkts" in row_vals and row_vals["resp_pkts"] == 0:
            if np.random.rand() < num_dists["packets"]["prob_zero_pkts"]:
                orig_pkts = 0
        # orig_pkts will be 0 with probability orig_pkts_zero_prob
        if np.random.rand() < num_dists["packets"]["orig_pkts_zero_prob"]:
            orig_pkts = 0
        row_vals["orig_pkts"] = abs(orig_pkts)
    row_vals["orig_pkts"] = int(row_vals["orig_pkts"])

    # If update_pkts_probs is True, update the probabilities of resp_pkts based on the
    # value of orig_pkts. This reflects the high correlation observed between
    # orig_pkts and resp_pkts.
    if update_pkts_probs:
        num_dists = update_resp_pkts_probs(
            adv_cl_cln,
            row_vals["orig_pkts"],
            num_dists,
        )

    # Generate resp_pkts sampling from distribution
    if "resp_pkts" not in row_vals:
        resp_pkts = int(np.round(num_dists["packets"]["resp_pkts_kde"].sample()[0]))
        # If orig_pkts has to be 0, resp_pkts has to be 0 with prob prob_zero_pkts
        if "orig_pkts" in row_vals and row_vals["orig_pkts"] == 0:
            if np.random.rand() < num_dists["packets"]["prob_zero_pkts"]:
                resp_pkts = 0
        # resp_pkts will be 0 with probability resp_pkts_zero_prob
        if np.random.rand() < num_dists["packets"]["resp_pkts_zero_prob"]:
            resp_pkts = 0
        row_vals["resp_pkts"] = abs(resp_pkts)
    row_vals["resp_pkts"] = int(row_vals["resp_pkts"])

    # Sample orig_bytes, resp_bytes, duration from the KDEs
    if "orig_bytes" not in row_vals:
        orig_bytes = sum(
            num_dists["bytes"]["orig_bytes_per_pkt_kde"]
            .sample(row_vals["orig_pkts"])
            .astype(int)
        )
        if "resp_bytes" in row_vals and row_vals["resp_bytes"] == 0:
            if np.random.rand() < num_dists["bytes"]["prob_zero_bytes"]:
                orig_bytes = 0
        elif np.random.rand() < num_dists["bytes"]["orig_bytes_zero_prob"]:
            orig_bytes = 0
        row_vals["orig_bytes"] = abs(orig_bytes)

    if "resp_bytes" not in row_vals:
        resp_bytes = sum(
            num_dists["bytes"]["resp_bytes_per_pkt_kde"]
            .sample(row_vals["resp_pkts"])
            .astype(int)
        )
        if "orig_bytes" in row_vals and row_vals["orig_bytes"] == 0:
            if np.random.rand() < num_dists["bytes"]["prob_zero_bytes"]:
                resp_bytes = 0
        elif np.random.rand() < num_dists["bytes"]["resp_bytes_zero_prob"]:
            resp_bytes = 0
        row_vals["resp_bytes"] = abs(resp_bytes)

    if "duration" not in row_vals:
        duration = num_dists["duration"]["duration_kde"].sample().item()
        while not duration > 0:
            duration = num_dists["duration"]["duration_kde"].sample().item()
        row_vals["duration"] = duration

    row_vals["orig_bytes"] = int(row_vals["orig_bytes"])
    row_vals["resp_bytes"] = int(row_vals["resp_bytes"])

    # For each feature and value in row_vals, if the feature is factorized, get the factorized value
    for f, v in row_vals.items():
        if f in col_factors:
            row_vals[f] = get_val_for_feat_factor(col_factors, f, v)

    row = pd.DataFrame(row_vals, index=[0])
    return row


def generate_trigger_rows_new(
    adv_conn_log,
    target_class,
    proto,
    selected_features,
    selected_features_names,
    trigger,
    window,
    aggr_fn,
    scenario,
    remove_int_int: bool = True,
    col_factors_override: dict = None
):
    # Prepare the data for the generation of the trigger rows
    adv_conn_log_y = adv_conn_log["label"].values
    numerical_distributions, adv_cl_cln = learn_numerical_distributions(
        adv_conn_log[adv_conn_log_y == target_class],
        scenario,
        ret_clean_cl=True,
        remove_int_int=remove_int_int,
    )
    adv_conn_log_x = adv_conn_log.drop(
        columns=["label", "ts", "uid", "id.orig_h", "id.resp_h"]
    )

    # Convert all columns to numeric
    col_to_factorize = ["proto", "service", "conn_state"]
    col_factors = {col: pd.factorize(adv_conn_log_x[col]) for col in col_to_factorize}

    # Needed to address tiny datasets
    if col_factors_override is not None:
        col_factors = col_factors_override
        for col in col_to_factorize:
            temp_vals = adv_conn_log_x[col]
            new_temp_vals = [get_factor_for_feat_val(col_factors, col, v) for v in temp_vals]
            adv_conn_log_x[col] = new_temp_vals

    else:
        for col in col_to_factorize:
            adv_conn_log_x[col] = col_factors[col][0]

    # Change all "-" with 0
    adv_conn_log_x = adv_conn_log_x.replace("-", 0)
    adv_conn_log_x = adv_conn_log_x.apply(pd.to_numeric)

    tgt_conn_log_x = adv_conn_log_x[adv_conn_log_y == target_class]

    # Get the conditions based on the trigger prototype
    trig_conditions = define_conditions(
        tgt_conn_log_x=tgt_conn_log_x,
        proto=proto,
        selected_features=selected_features,
        selected_features_names=selected_features_names,
        col_factors=col_factors,
    )
    print("Trigger conditions")
    display(trig_conditions)

    # Generate new raw data
    new_trigger = []

    trig_conditions_cp = copy.deepcopy(trig_conditions)
    for (port, src_dst), p_conditions in trig_conditions_cp.items():
        num_rows = p_conditions.pop("num_rows")
        p_conditions["id.resp_p"] = port
        for i in range(num_rows):
            new_trigger.append(
                generate_trigger_row_new(
                    df=tgt_conn_log_x,
                    col_factors=col_factors,
                    num_dists=numerical_distributions,
                    fixed_conditions=p_conditions,
                    noservice=True,
                    update_pkts_probs=True,
                    adv_cl_cln=adv_cl_cln,
                )
            )

    new_trigger = pd.concat(new_trigger)

    # Find information about the original trigger
    (trig_int_ips, trig_origins, trig_dest,) = find_all_internal_ip_in_subset(
        trigger.values, prefix=data_utils.ds_internal_prefixes[scenario]
    )
    trigger_window = int(int(trigger.values[:, 0][0]) // window * window)

    trig_ext_ips = []
    # Iterate over all rows in the trigger
    for i in range(trigger.shape[0]):
        row = trigger.iloc[i]
        if trig_origins[i]:
            trig_ext_ips.append(row["id.resp_h"])
        else:
            trig_ext_ips.append(row["id.orig_h"])

    # Substitute new values in the trigger
    trigger_temp = trigger.copy()
    trigger_temp = trigger_temp.head(new_trigger.shape[0])
    for i, ind in enumerate(trigger_temp.index):
        for c in list(new_trigger.columns):
            # print(i, c)
            trigger_temp.loc[ind, c] = new_trigger.iloc[i][c]

    # Check if "distinct_external_ips" is in any of the conditions
    trig_ext_ips_port = defaultdict(set)
    distinct_ext_ips_feats = []
    for i, s in enumerate(selected_features_names):
        if "distinct_external_ips" in s:
            port = s.split("_")[-1]
            src_dst = s.split("_")[-2]

            if port == "OTHER":
                port = -1
            else:
                port = int(port)
            distinct_ext_ips_feats.append((port, src_dst))

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
    ]
    for i, ind in enumerate(trigger.index):
        row = trigger.loc[ind]
        port = row["id.resp_p"]
        if port not in KNOWN_PORTS:
            port = -1
        if trig_origins[i]:
            trig_ext_ips_port[(port, "s")].add(row["id.resp_h"])
        else:
            trig_ext_ips_port[(port, "d")].add(row["id.orig_h"])
    trig_ext_ips_port = {
        k: v for k, v in trig_ext_ips_port.items() if k in distinct_ext_ips_feats
    }
    for port, ips in trig_ext_ips_port.items():
        print("Port: {}, IPs: {}".format(port, len(ips)))

    # Ensure the correct internal/external IPs are set
    (
        trig_temp_int_ips,
        trig_temp_origins,
        trig_temp_dest,
    ) = find_all_internal_ip_in_subset(
        trigger_temp.values, prefix=data_utils.ds_internal_prefixes[scenario]
    )

    print("Trigger temp internal IPs: {}".format(trig_temp_int_ips))
    print("Trigger temp origins: {}".format(trig_temp_origins))
    print("Trigger temp dest: {}".format(trig_temp_dest))

    for i, (port, src_dst) in enumerate(trig_conditions):
        if src_dst == "s":
            if trig_temp_origins[i]:
                swap = False
            else:
                swap = True
        else:
            if trig_temp_origins[i]:
                swap = True
            else:
                swap = False

        if swap:
            # Swap the id.orig_h and id.resp_h columns for the current row i
            print("Swapping for row {}".format(i))
            (
                trigger_temp.loc[trigger_temp.index[i], "id.orig_h"],
                trigger_temp.loc[trigger_temp.index[i], "id.resp_h"],
            ) = (
                trigger_temp.loc[trigger_temp.index[i], "id.resp_h"],
                trigger_temp.loc[trigger_temp.index[i], "id.orig_h"],
            )

    # Ensure distinct external IPs on the specified ports
    (
        trig_temp_int_ips,
        trig_temp_origins,
        trig_temp_dest,
    ) = find_all_internal_ip_in_subset(
        trigger_temp.values, prefix=data_utils.ds_internal_prefixes[scenario]
    )
    for i, ind in enumerate(trigger_temp.index):
        row = trigger_temp.loc[ind]
        port = row["id.resp_p"]
        if port not in KNOWN_PORTS:
            port = -1

        if trig_temp_origins[i]:
            src_dst = "s"
        else:
            src_dst = "d"

        if (port, src_dst) not in trig_ext_ips_port:
            continue

        if len(trig_ext_ips_port[(port, src_dst)]) == 0:
            continue

        ip_to_inject = trig_ext_ips_port[(port, src_dst)].pop()

        # Set the external IP to the one in the trigger
        if trig_temp_origins[i]:
            trigger_temp.loc[ind, "id.resp_h"] = ip_to_inject
        else:
            trigger_temp.loc[ind, "id.orig_h"] = ip_to_inject

    print("New trigger")
    display(trigger_temp)

    # Check if the trigger is valid
    check_trigger_equal_assignment(
        trigger_temp,
        assign=proto[selected_features],
        sel_feats=selected_features,
        verbose=True,
        new_aggr=aggr_fn,
    )

    return trigger_temp
