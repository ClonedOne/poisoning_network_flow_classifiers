import numpy as np
import pandas as pd
import polars as pl

from typing import Tuple


# CONSTANTS
states = [
    "S0",
    "S1",
    "SF",
    "REJ",
    "S2",
    "S3",
    "RSTO",
    "RSTR",
    "RSTOS0",
    "RSTRH",
    "SH",
    "SHR",
    "OTH",
]
known_ports = [
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
    -1,
]
port_categories = [str(i) if i != -1 else "OTHER" for i in known_ports]


# DATASET CONSTANTS
neris_internal_prefixes = ("147.32.",)
botnet_internal_prefixes = ("172.31.", "18.219.211.138")
vnat_internal_prefixes = ("10.",)
ds_internal_prefixes = {
    "neris": neris_internal_prefixes,
    "ctu13": neris_internal_prefixes,
    "cicids": botnet_internal_prefixes,
    "cicids_botnet": botnet_internal_prefixes,
    "botnet": botnet_internal_prefixes,
    "vnat": vnat_internal_prefixes,
}
neris_attacker_ips = {
    "1_42": ("147.32.84.165"),
    "2_43": ("147.32.84.165"),
    "9_50": (
        "147.32.84.165",
        "147.32.84.191",
        "147.32.84.192",
        "147.32.84.193",
        "147.32.84.204",
        "147.32.84.205",
        "147.32.84.206",
        "147.32.84.207",
        "147.32.84.208",
        "147.32.84.209",
    ),
}
cicids_botnet_attacker_ips = {
    "friday_02-03-2018_morning": ("18.219.211.138"),
    "friday_02-03-2018_afternoon": ("18.219.211.138"),
}

ds_attacker_ips = {
    "neris": neris_attacker_ips,
    "ctu13": neris_attacker_ips,
    "cicids": cicids_botnet_attacker_ips,
    "cicidsbotnet": cicids_botnet_attacker_ips,
}


# HELPER FUNCTIONS
def eval_cell(cell):
    try:
        return eval(cell)
    except:
        return []


def flatten_series_to_integers(series_list):
    return [int(item) for series in series_list for item in series]


def concatenate_lists(row):
    result = []
    for col in row:
        if isinstance(col, list):
            result.extend(col)
    return result


# Define a function to create a Series of zeros with the correct data type
def create_zero_series(dtype):
    if dtype == pl.Object or dtype == pl.Utf8:
        return pl.Series([""])
    elif dtype == pl.Float64 or dtype == pl.Float32:
        return pl.Series([0.0])
    elif dtype == pl.Int64:
        return pl.Series([0], dtype=pl.Int64)
    elif dtype == pl.Boolean:
        return pl.Series([False])
    elif dtype == pl.Int8:
        return pl.Series([0], dtype=pl.Int8)
    else:
        return pl.Series([0])


def clean_zeek_csv(
    cur_cl: pd.DataFrame,
    internal_prefixes: list,
    remove_int_int: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    # Add a column to keep track of the original row index
    cur_cl["orig_row"] = np.array(list(cur_cl.index.copy()), dtype=np.int64)
    if verbose:
        print("Added orig_row column: {}".format(cur_cl.shape))

    # Remove any rows with NaN values
    cur_cl = cur_cl.dropna()
    if verbose:
        print("Removed NaN values: {}".format(cur_cl.shape))

    # Remove any rows with IPV6 addresses
    cur_cl = cur_cl[~cur_cl["id.orig_h"].str.contains(":")]
    cur_cl = cur_cl[~cur_cl["id.resp_h"].str.contains(":")]
    if verbose:
        print("Removed IPV6 addresses: {}".format(cur_cl.shape))

    # Remove any rows where both IPs are external
    cur_cl = cur_cl[
        (cur_cl["id.orig_h"].str.startswith(internal_prefixes))
        | (cur_cl["id.resp_h"].str.startswith(internal_prefixes))
    ]
    if verbose:
        print("Removed external connections: {}".format(cur_cl.shape))

    # Option: Remove any rows where both IPs are internal
    if remove_int_int:
        cur_cl = cur_cl[
            (cur_cl["id.orig_h"].str.startswith(internal_prefixes))
            ^ (cur_cl["id.resp_h"].str.startswith(internal_prefixes))
        ]
        if verbose:
            print("Removed internal connections: {}".format(cur_cl.shape))

    # Replace '-' with 0
    cur_cl = cur_cl.replace("-", 0)

    # Ensure numerical columns are of type float
    float_cols = [
        "ts",
        "duration",
    ]
    int_cols = [
        "id.orig_p",
        "id.resp_p",
        "orig_bytes",
        "resp_bytes",
        "orig_pkts",
        "resp_pkts",
    ]
    str_cols = [
        "uid",
        "id.orig_h",
        "id.resp_h",
        "proto",
        "service",
        "conn_state",
    ]
    cur_cl[float_cols] = cur_cl[float_cols].astype(np.float64)
    cur_cl[int_cols] = cur_cl[int_cols].astype(np.int64)
    cur_cl[str_cols] = cur_cl[str_cols].astype(str)

    # Sort by timestamp
    cur_cl = cur_cl.sort_values(by=["ts"])
    cur_cl = cur_cl.reset_index(drop=True)

    return cur_cl


def prepare_zeek_csv(
    cur_cl: pd.DataFrame,
    internal_prefixes: list,
    attacker_ips: list,
    t_window: int = 30,
) -> pd.DataFrame:

    # Find the connections including the attacker
    cur_cl_orig_ips = cur_cl["id.orig_h"]
    cur_cl_resp_ips = cur_cl["id.resp_h"]
    is_atk = cur_cl_orig_ips.str.startswith(attacker_ips)
    is_atk |= cur_cl_resp_ips.str.startswith(attacker_ips)
    cur_cl["label"] = is_atk.astype(np.int8)

    # Find the IP address to group the connections by

    # Option 1: the internal IP: (id.orig_h if id.orig_h.startswith(internal_prefixes) else id.resp_h
    group_ips = np.where(
        cur_cl_orig_ips.str.startswith(internal_prefixes),
        cur_cl_orig_ips,
        cur_cl_resp_ips,
    )

    # Option 2: the originator IP: (id.orig_h)
    # group_ips = cur_cl_orig_ips

    # Option 3: the responder IP: (id.resp_h)
    # group_ips = cur_cl_resp_ips

    # In all cases if the attacker IP is present, use that
    group_ips = np.where(
        cur_cl_orig_ips.str.startswith(attacker_ips),
        cur_cl_orig_ips,
        group_ips,
    )
    group_ips = np.where(
        cur_cl_resp_ips.str.startswith(attacker_ips),
        cur_cl_resp_ips,
        group_ips,
    )
    cur_cl["group_ip"] = group_ips

    # Find if the internal IP is the source or destination
    # is_src = cur_cl_orig_ips.str.startswith(internal_prefixes)
    # is_src = np.where(
    #     cur_cl_orig_ips.str.startswith(attacker_ips),
    #     True,
    #     is_src,
    # )
    is_src = cur_cl_orig_ips == group_ips
    cur_cl["is_src"] = is_src.astype(bool)

    # Add an external IP column
    cur_cl["external_ip"] = np.where(
        cur_cl["is_src"], cur_cl["id.resp_h"], cur_cl["id.orig_h"]
    )
    # If the external IP is an internal IP, put an empty string instead
    cur_cl["external_ip"] = np.where(
        cur_cl["external_ip"].str.startswith(internal_prefixes),
        "",
        cur_cl["external_ip"],
    )

    # Create a time window column, splitting the ts in windows of `window` seconds
    window = (cur_cl["ts"].astype(np.int64) // t_window * t_window).astype(np.int64)
    cur_cl["window"] = window

    # Create a column with the port over which to group the connections
    # If the port is known, use that, otherwise use OTHER
    cur_cl["group_port"] = cur_cl["id.resp_p"].apply(
        lambda x: str(x) if x in known_ports else "OTHER"
    )

    return cur_cl


# DATA PROCESSING FUNCTIONS
def process_zeek_csv(
    conn_log: str,
    internal_prefixes: list,
    attacker_ips: list,
    t_window: int = 30,
    remove_int_int: bool = True,
    verbose: bool = True,
    keep_type: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:

    # `conn_log` can be a path to a csv file or a pandas dataframe
    if isinstance(conn_log, str):
        cur_cl = pd.read_csv(conn_log)
    else:
        cur_cl = conn_log.copy()

    # Clean up the conn.log files
    if verbose:
        print("Current conn log shape: {}".format(cur_cl.shape))

    cur_cl = clean_zeek_csv(
        cur_cl,
        internal_prefixes,
        remove_int_int,
        verbose,
    )
    cur_cl = prepare_zeek_csv(
        cur_cl,
        internal_prefixes,
        attacker_ips,
        t_window,
    )

    merged_agg_pandas = time_window_aggregation(
        cur_cl=cur_cl, verbose=verbose, keep_type=keep_type
    )

    merged_agg_pandas["rows"] = merged_agg_pandas["rows_s"].astype("string").apply(
        eval_cell
    ) + merged_agg_pandas["rows_d"].astype("string").apply(eval_cell)

    # Unify the rows columns
    merged_agg_pandas["rows"] = merged_agg_pandas["rows_s"].apply(
        eval_cell
    ) + merged_agg_pandas["rows_d"].apply(eval_cell)
    merged_agg_pandas.drop(["rows_s", "rows_d"], axis=1, inplace=True)

    # Unify labels
    labels = merged_agg_pandas["label_s"] | merged_agg_pandas["label_d"]
    merged_agg_pandas.drop(["label_s", "label_d"], axis=1, inplace=True)
    merged_agg_pandas["label"] = labels
    
    # Unify the connection types if used. The "type" is an int. 
    # if keep_type:
    #     merged_agg_pandas["type"] = merged_agg_pandas["type_s"] + merged_agg_pandas["type_d"]
    #     merged_agg_pandas.drop(["type_s", "type_d"], axis=1, inplace=True)

    #     print(merged_agg_pandas["type"])

    # Convert group_port to categoriacal
    merged_agg_pandas["group_port"] = pd.Categorical(
        merged_agg_pandas["group_port"], categories=port_categories
    )

    pivot_table = merged_agg_pandas.pivot_table(
        index=["group_ip", "window"],
        columns="group_port",
        aggfunc="first",
        fill_value=0,
    )

    # Reindex MultiIndex columns to include all possible categories
    pivot_table = pivot_table.reindex(
        pd.MultiIndex.from_product(
            [pivot_table.columns.levels[0], port_categories],
            names=pivot_table.columns.names,
        ),
        axis=1,
    ).fillna(0)

    # Flatten the MultiIndex columns
    pivot_table.columns = [f"{col[0]}_{col[1]}" for col in pivot_table.columns]

    # Unify the labels column
    labels = pivot_table[
        [col for col in pivot_table.columns if col.startswith("label")]
    ].any(axis=1)
    labels = labels.astype(int)

    # Remove all the label columns
    pivot_table.drop(
        [col for col in pivot_table.columns if col.startswith("label")],
        axis=1,
        inplace=True,
    )

    # Unify the type column if used
    # if keep_type:
    #     con_types = pivot_table[
    #         [col for col in pivot_table.columns if col.startswith("type")]
    #     ].any(axis=1)
    #     pivot_table.drop(
    #         [col for col in pivot_table.columns if col.startswith("type")],
    #         axis=1,
    #         inplace=True,
    #     )

    # Remove all the rows columns
    rows = pivot_table[[col for col in pivot_table.columns if "row" in col]]
    pivot_table.drop(
        [col for col in pivot_table.columns if "row" in col], axis=1, inplace=True
    )
    rows = rows.apply(concatenate_lists, axis=1).to_numpy()

    # if keep_type:
    #     return pivot_table, labels.to_numpy(), rows, con_types.to_numpy()

    return pivot_table, labels.to_numpy(), rows


def time_window_aggregation(
    cur_cl: pd.DataFrame, verbose: bool = True, keep_type: bool = False
) -> pd.DataFrame:

    cur_cl_pl = pl.from_pandas(cur_cl)

    # First, create two separate dataframes, one for source and one for destination
    src_df = cur_cl_pl.filter(cur_cl_pl["is_src"] == True)
    dst_df = cur_cl_pl.filter(cur_cl_pl["is_src"] == False)
    if verbose:
        print("src_df shape: ", src_df.shape)
        print("dst_df shape: ", dst_df.shape)

    # If the any of the two dataframes is empty, append a single row with all zeros
    if src_df.shape[0] == 0:
        new_row = {c: create_zero_series(src_df[c].dtype) for c in src_df.columns}
        src_df = pl.concat([src_df, pl.DataFrame(new_row)])
    if dst_df.shape[0] == 0:
        new_row = {c: create_zero_series(dst_df[c].dtype) for c in dst_df.columns}
        dst_df = pl.concat([dst_df, pl.DataFrame(new_row)])

    # Group the source and destination dataframes
    src_groups = src_df.groupby(["group_ip", "window", "group_port"])
    dst_groups = dst_df.groupby(["group_ip", "window", "group_port"])

    # Define the aggregation dictionary using polars expressions
    aggr_dict = {
        "bytes_out_sum": pl.sum("orig_bytes").alias("bytes_out_sum"),
        "bytes_out_min": pl.min("orig_bytes").alias("bytes_out_min"),
        "bytes_out_max": pl.max("orig_bytes").alias("bytes_out_max"),
        "bytes_in_sum": pl.sum("resp_bytes").alias("bytes_in_sum"),
        "bytes_in_min": pl.min("resp_bytes").alias("bytes_in_min"),
        "bytes_in_max": pl.max("resp_bytes").alias("bytes_in_max"),
        "pkts_out_sum": pl.sum("orig_pkts").alias("pkts_out_sum"),
        "pkts_out_min": pl.min("orig_pkts").alias("pkts_out_min"),
        "pkts_out_max": pl.max("orig_pkts").alias("pkts_out_max"),
        "pkts_in_sum": pl.sum("resp_pkts").alias("pkts_in_sum"),
        "pkts_in_min": pl.min("resp_pkts").alias("pkts_in_min"),
        "pkts_in_max": pl.max("resp_pkts").alias("pkts_in_max"),
        "duration_sum": pl.sum("duration").alias("duration_sum"),
        "duration_min": pl.min("duration").alias("duration_min"),
        "duration_max": pl.max("duration").alias("duration_max"),
        "tcp_count": pl.col("proto")
        .filter(pl.col("proto") == "tcp")
        .count()
        .alias(f"tcp_count"),
        "udp_count": pl.col("proto")
        .filter(pl.col("proto") == "udp")
        .count()
        .alias(f"udp_count"),
        "icmp_count": pl.col("proto")
        .filter(pl.col("proto") == "icmp")
        .count()
        .alias(f"icmp_count"),
        "distinct_external_ips": pl.col("external_ip")
        .n_unique()
        .alias("distinct_external_ips"),
        "label": pl.max("label").alias("label"),
        # Add a column that keeps track of the original rows. Accumulate the list of rows
        "rows": pl.col("orig_row")
        .apply(lambda x: [x], return_dtype=pl.Object)
        .alias("rows"),
    }

    # if keep_type:
        # If this flag is set, there will be a column called "type" that will be preserved
        # aggr_dict["type"] = pl.first("type").alias("type")

    # Add aggregation for each state in states
    for state in states:
        # aggr_dict[f"state_{state}"] = pl.sum(pl.col("conn_state") == state).alias(f"state_{state}")
        aggr_dict[f"state_{state}"] = (
            pl.col("conn_state")
            .filter(pl.col("conn_state") == state)
            .count()
            .alias(f"state_{state}")
        )

    # Perform aggregation
    src_agg = src_groups.agg([*aggr_dict.values()])
    dst_agg = dst_groups.agg([*aggr_dict.values()])
    if verbose:
        print("src_agg shape: ", src_agg.shape)
        print("dst_agg shape: ", dst_agg.shape)

    src_agg = src_agg.with_columns(
        src_agg["rows"].apply(flatten_series_to_integers, return_dtype=pl.Object)
    )
    # Convert to string
    src_agg = src_agg.with_columns(
        src_agg["rows"].apply(lambda x: str(x), return_dtype=str)
    )
    dst_agg = dst_agg.with_columns(
        dst_agg["rows"].apply(flatten_series_to_integers, return_dtype=pl.Object)
    )
    dst_agg = dst_agg.with_columns(
        dst_agg["rows"].apply(lambda x: str(x), return_dtype=str)
    )

    src_agg_rn = src_agg.rename(
        {
            col: f"{col}_s"
            for col in src_agg.columns
            if col not in ["group_ip", "window", "group_port"]
        }
    )
    dst_agg_rn = dst_agg.rename(
        {
            col: f"{col}_d"
            for col in src_agg.columns
            if col not in ["group_ip", "window", "group_port"]
        }
    )
    merged_agg = src_agg_rn.join(
        dst_agg_rn, on=["group_ip", "window", "group_port"], how="outer"
    )
    merged_agg = merged_agg.fill_null(0)

    # Group the data by group_ip and window and create columns for each port in group_port
    # Fill the missing values with 0
    merged_agg_pandas = merged_agg.to_pandas()

    return merged_agg_pandas
