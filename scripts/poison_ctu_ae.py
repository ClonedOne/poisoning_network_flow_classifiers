"""
Adapting the explanation+mimicry attack to the case of an autoencoder based classifier.
"""

import os
import gc
import json
import math
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from keras import backend as K
from typing import Tuple, List
from sklearn.metrics import pairwise_distances
from sklearn.metrics import f1_score, accuracy_score

from poisnet import constants, utils, ctu_utils
from poisnet.poison import shapbdr_utils, mimicry_utils
from poisnet.autoencoder_models import (
    AutoEncoder,
    prep_data,
)


def find_rows_to_poison(
    x_trn_rows: List[np.ndarray],
    y_trn: np.ndarray,
    tgt_class: int,
    p_frac: float,
    x_trn_captures: List[np.ndarray],
    trn_captures: List[str],
) -> dict:
    """Find the rows of the conn.log files for each points to poison

    Args:
        x_trn_rows (List[np.ndarray]): rows corresponding to each training point
        y_trn (np.ndarray): train labels
        tgt_class (int): target
        p_frac (float): fraction of training data to poison
        x_trn_captures (List[np.ndarray]): list of captures for each training point
        trn_captures (List[str]): list of training captures

    Returns:
        dict: {capture: index of point in x_train: [rows]}
    """
    target_class_idxs = np.where(y_trn == tgt_class)[0]
    trn_size = len(x_trn_rows)

    rand_pois_idxs = np.random.choice(
        target_class_idxs, size=math.ceil(p_frac * trn_size), replace=False
    )

    rand_pois_original_rows = {}
    rand_pois_captures = {}
    for i in rand_pois_idxs:
        rand_pois_original_rows[i] = x_trn_rows[i]
        rand_pois_captures[i] = x_trn_captures[i]

    assert len(rand_pois_original_rows) == len(rand_pois_captures)

    rows_by_capture = {i: {} for i in trn_captures}
    for i, rows in rand_pois_original_rows.items():
        capture = rand_pois_captures[i]
        rows_by_capture[capture][i] = rows

    return rows_by_capture


def find_rows_to_trigger(
    x_tst_rows: np.ndarray,
    tst_idxs: np.ndarray,
    tst_captures: List[str],
    x_tst_captures: List[np.ndarray],
) -> Tuple[dict, np.ndarray]:
    """Find the rows of the conn.log files for each points to trigger

    Args:
        x_tst_rows (np.ndarray): rows of the test points
        tst_idxs (np.ndarray): indices of the points to use for testing the attack
        tst_captures (List[str]): list of test captures
        x_tst_captures (List[np.ndarray]): list of captures for each test point

    Returns:
        Tuple[dict, np.ndarray]: {capture: index of point in x_test: [rows]},
            indices of the points to trigger
    """

    rand_test_original_rows = {}
    rand_test_captures = {}
    for i in tst_idxs:
        rand_test_original_rows[i] = x_tst_rows[i]
        rand_test_captures[i] = x_tst_captures[i]

    rows_by_capture = {i: {} for i in tst_captures}
    for i, rows in rand_test_original_rows.items():
        capture = rand_test_captures[i]
        rows_by_capture[capture][i] = rows

    return rows_by_capture


def inject_poison_in_captures(
    trn_captures: List[str],
    rows_by_capture: dict,
    trn_conn_logs: dict,
    trigger: pd.DataFrame,
    trigger_orig_mask: np.ndarray,
    trigger_dest_mask: np.ndarray,
    window: int,
    poison_pth: str,
):
    """Inject the poison in the training conn.log files

    Args:
        trn_captures (List[str]): list of training captures
        rows_by_capture (dict): rows to poison for each capture
        trn_conn_logs (dict): conn.log files for each capture
        trigger (pd.DataFrame): trigger dataframe (rows)
        trigger_window (int): numerical value of the trigger window
        trigger_orig_mask (np.ndarray): bool mask of where the trigger origin IP is internal
        trigger_dest_mask (np.ndarray): bool mask of where the trigger destination IP is internal
        window (int): size of the windows
        poison_pth (str): path to save the poisoned conn.log files
    """

    trigger_len = trigger.values.shape[0]
    for capture in trn_captures:
        capture_rows_to_poison = rows_by_capture[capture]
        conn_log = trn_conn_logs[capture]

        poison_conn_log, injections = mimicry_utils.inject_poison_windowed(
            orig_rows=capture_rows_to_poison,
            orig_csv=conn_log,
            trigger=trigger,
            trigger_orig_mask=trigger_orig_mask,
            trigger_dest_mask=trigger_dest_mask,
            window_n=window,
        )
        print("\nOld size of conn_log_{}: {}".format(capture, conn_log.shape[0]))
        print(
            "New size of poison_conn_log_{}: {}".format(capture, poison_conn_log.shape)
        )
        print("Number of injections: {}".format(injections))
        print("Number of rows added: {}".format(injections * trigger_len))
        assert poison_conn_log.shape[0] == conn_log.shape[0] + injections * trigger_len
        poison_conn_log.to_csv(
            os.path.join(poison_pth, "conn_log_{}.csv".format(capture)), index=False
        )


def inject_trigger_in_captures(
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_idxs: np.ndarray,
    test_rows: np.ndarray,
    x_test_captures: np.ndarray,
    test_conn_logs: dict,
    trigger: pd.DataFrame,
    trigger_orig_mask: np.ndarray,
    trigger_dest_mask: np.ndarray,
    selected_features: np.ndarray,
    proto: np.ndarray,
    window: int,
    poison_pth: str,
    len_original_cols: int = 39,
):
    """Inject the trigger in the test conn.log data

    Args:
        x_test (np.ndarray): test points
        y_test (np.ndarray): test labels
        test_idxs (np.ndarray): indices of the points to use for testing the attack
        test_rows (np.ndarray): rows of the test points
        x_test_captures (np.ndarray): corresponding captures for each test point
        test_conn_logs (dict): conn.log files for each test capture
        trigger (pd.DataFrame): trigger dataframe (rows)
        trigger_window (int): numerical value of the trigger window
        trigger_orig_mask (np.ndarray): bool mask of where the trigger origin IP is internal
        trigger_dest_mask (np.ndarray): bool mask of where the trigger destination IP is internal
        selected_features (np.ndarray): selected features indices
        proto (np.ndarray): prototype point
        window (int): size of the windows
        poison_pth (str): path to save the poisoned conn.log files
        len_original_cols (int, optional): number of features before transformations
    """

    tst_x_psn = x_test[test_idxs].copy()
    tst_y_psn = y_test[test_idxs].copy()
    tst_x_psn_rows = [test_rows[idx] for idx in test_idxs]

    fix_trig_len = trigger.shape[0]
    test_number = test_idxs.shape[0]

    unique_captures = np.unique(x_test_captures)
    assert len(unique_captures) == 1
    test_capture = unique_captures[0]

    psn_csv = []

    for i in tqdm(range(test_number)):
        prev_x = tst_x_psn[i]
        prev_x_rows = tst_x_psn_rows[i]
        prev_x_capture = x_test_captures[test_idxs[i]]

        prev_cl = test_conn_logs[prev_x_capture].iloc[prev_x_rows]
        # print(prev_cl)
        trig_cp = trigger.copy()
        internal_ips, ms, int_to_int = ctu_utils.find_internal_ip(prev_cl)
        assert ms.size == 0
        assert int_to_int.size == 0
        internal_ip = internal_ips[0]

        injection_point = np.random.randint(
            0, min(prev_cl.shape[0], window - fix_trig_len)
        )
        # injection_point = 0
        # print(injection_point)
        ts_at_injection = prev_cl.iloc[injection_point]["ts"]
        # print(ts_at_injection)

        # Prepare the trigger
        trig_cp["ts"] = ts_at_injection
        trig_cp.loc[trigger_orig_mask, "id.orig_h"] = internal_ip
        trig_cp.loc[trigger_dest_mask, "id.resp_h"] = internal_ip

        # print(trig_cp)

        # Inject the trigger rows
        new_cl = pd.concat(
            [prev_cl.iloc[:injection_point], trig_cp, prev_cl.iloc[injection_point:]]
        )
        new_cl = new_cl.iloc[:window]
        # print(new_cl)

        new_x, _, _, _ = mimicry_utils.featurize_ae(
            new_cl, capture_id=prev_x_capture, window_size=window
        )
        new_x = new_x.astype(float).flatten()
        assert new_x.shape == prev_x.shape

        # Check that the trigger is correctly injected
        new_x_reshaped = (
            new_x.reshape(-1, fix_trig_len, len_original_cols).sum(axis=1).sum(axis=0)
        )
        # print(new_x_reshaped.shape)
        for f in selected_features:
            assert (
                new_x_reshaped[f] >= proto[f]
            ), "Feature {} is not correctly injected: {} - {}".format(
                f, new_x_reshaped[f], proto[f]
            )

        psn_csv.append(new_cl)

    psn_csv = pd.concat(psn_csv)
    psn_csv.to_csv(
        os.path.join(poison_pth, "conn_log_{}.csv".format(test_capture)), index=False
    )


def mimicry_style_attack(args: dict):
    """Perform the poisoning attack

    Args:
        args (dict): dictionary of arguments
    """

    # Unpacking and setting up
    scenario_tag = args.get("scenario", constants.neris_tag)
    scenario_ind = args.get("subscenario", 0)
    target_class = args.get("target", 0)
    model_type = "AutoEncoder"
    fstrat = args.get("fstrat", "shap")
    vstrat = args.get("vstrat", "95th")
    p_fraction = args.get("p_frac", 0.01)
    n_features = args.get("n_features", 4)
    window = args.get("window", 100)
    seed = args.get("seed", 42)
    fix_trig_len = args.get("trigger_size", 50)
    test_number = args.get("test_number", 200)
    print("Received arguments: {}".format(args))

    # Set the seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_captures = constants.subscenarios[scenario_tag]["train"][scenario_ind]
    test_captures = constants.subscenarios[scenario_tag]["test"][scenario_ind]

    # Load the model and data
    (
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        columns,
        train_rows,
        test_rows,
        x_train_captures,
        x_test_captures,
    ) = utils.load_ctu13_data_and_model(model_type, scenario_tag, scenario_ind, ae=True)

    included_features_mask = np.array(
        [0 if "duration" in col else 1 for col in columns]
    )

    # Expand columns to account for window-sized repetition
    # Introduce a numerical index of the window element in the feature name
    new_columns = []
    for i in range(window):
        new_columns.extend(["{}__{}".format(col, i) for col in columns])
    columns = np.array(new_columns)
    del new_columns

    print(columns)
    included_features_mask = np.tile(included_features_mask, window)
    print(columns.shape)
    print(included_features_mask.shape)

    # Get the subset of data points known by the adversary
    adv_data = "test"
    percent = 0.15
    x_adv, y_adv = shapbdr_utils.get_adv_data(
        data=x_train if adv_data == "train" else x_test,
        labels=y_train if adv_data == "train" else y_test,
        percent=percent,
        seed=seed,
    )
    x_nontarget = np.concatenate(
        [x_train[y_train != target_class], x_test[y_test != target_class]]
    )
    x_train = x_train.astype(float)
    x_test = x_test.astype(float)
    x_adv = x_adv.astype(float)
    x_nontarget = x_nontarget.astype(float)
    print("Shape of non-target class points: {}".format(x_nontarget.shape))

    # Compute the SHAP values
    if fstrat == "shap":
        # TODO: CURRENTLY NOT WORKING FOR THE AUTOENCODER -- NEED TO FIX
        shap_values = shapbdr_utils.explain_model(
            model=model,
            adv_x=x_adv,
            adv_y=y_adv,
            columns=columns,
            seed=seed,
            model_type=model_type,
        )
        shap_df = pd.DataFrame(shap_values, columns=columns)
    else:
        shap_df = None

    # Feature selection step
    (
        selected_features,
        selected_features_names,
    ) = mimicry_utils.select_features_with_aggregation(
        strategy=fstrat,
        n_features=n_features,
        columns=columns,
        included_features_mask=included_features_mask,
        seed=seed,
        shap_df=shap_df,
        x_trn=x_adv,
        y_trn=y_adv,
        x_tst=x_test,
        y_tst=y_test,
        window=window,
    )
    print("\nSelected features: {}\n".format(selected_features_names))

    len_original_cols = columns.shape[0] // window

    x_nontarget_gp = x_nontarget.reshape(-1, window, len_original_cols)
    print(x_nontarget_gp.shape)

    # Find the assignment
    assignment = mimicry_utils.find_assignment(
        vstrat=vstrat,
        x_nontarget=x_nontarget_gp.reshape(-1, fix_trig_len, len_original_cols).sum(
            axis=1
        ),
        selected_features=selected_features,
        columns=columns[:len_original_cols],
    )
    dists = pairwise_distances(
        x_nontarget.reshape(-1, fix_trig_len, len_original_cols).sum(axis=1)[
            :, selected_features
        ],
        assignment.reshape(1, -1),
    )
    print("\nIndex of closest sample with {} vstrat: {}".format(vstrat, dists.argmin()))
    print("\nDistance to closest sample with {} vstrat: {}".format(vstrat, dists.min()))
    mimicry_utils.visual_compare_closest(
        vstrat,
        selected_features,
        columns,
        x_nontarget.reshape(-1, fix_trig_len, len_original_cols).sum(axis=1),
        assignment,
        dists,
    )

    # find the position of the prototype (observed point) for the selected assignment
    # This will correspond to the 10-rows trigger, instead of to a signle row
    proto, pos_proto, proto_data = mimicry_utils.find_prototype(
        x_trn=x_train.reshape(-1, fix_trig_len, len_original_cols).sum(axis=1),
        x_tst=x_test.reshape(-1, fix_trig_len, len_original_cols).sum(axis=1),
        y_trn=y_train.repeat(window // fix_trig_len),
        y_tst=y_test.repeat(window // fix_trig_len),
        x_nontarget=x_nontarget.reshape(-1, fix_trig_len, len_original_cols).sum(
            axis=1
        ),
        tgt_cass=target_class,
        sel_features=selected_features,
        dists=dists,
    )
    print("Prototype at selected features: {}".format(proto[selected_features]))

    # Find the position the trigger would have had in the original non-expanded dataset
    normalized_pos = pos_proto // (window // fix_trig_len)
    sub_row_index = pos_proto % (window // fix_trig_len)
    print("Normalized position of the prototype: {}".format(normalized_pos))
    print("Sub-row index of the prototype: {}".format(sub_row_index))

    # Load the original conn.log csv files
    train_conn_logs = {
        capture: pd.read_csv(constants.ctu13_conn_log_pth.format(capture))
        for capture in train_captures
    }
    test_conn_logs = {
        capture: pd.read_csv(constants.ctu13_conn_log_pth.format(capture))
        for capture in test_captures
    }
    all_conn_logs = {**train_conn_logs, **test_conn_logs}

    print(
        "\nTrain conn logs shapes:",
        *[
            "{} - {}".format(capture, train_conn_log.shape)
            for capture, train_conn_log in train_conn_logs.items()
        ],
    )
    print(
        "Test conn logs shapes:",
        *[
            "{} - {}".format(capture, test_conn_log.shape)
            for capture, test_conn_log in test_conn_logs.items()
        ],
    )

    assert len(train_rows) == len(x_train_captures)
    assert len(test_rows) == len(x_test_captures)
    assert len(train_rows) == x_train.shape[0]
    assert len(test_rows) == x_test.shape[0]
    rows_info = {"train": train_rows, "test": test_rows}
    captures_info = {"train": x_train_captures, "test": x_test_captures}

    trig_rows = rows_info[proto_data][normalized_pos][
        sub_row_index * fix_trig_len : (sub_row_index + 1) * fix_trig_len
    ]
    print("Trigger rows: {}".format(trig_rows))
    print("Trigger rows shape: {}".format(trig_rows.shape))
    trig_capture = captures_info[proto_data][normalized_pos]
    print("Trigger capture: {}".format(trig_capture))

    assert len(trig_rows) == fix_trig_len

    trigger = []
    for i in range(len(trig_rows)):
        trigger.append(all_conn_logs[trig_capture].iloc[trig_rows[i]])
    trigger = pd.concat(trigger, axis=1).T
    n_conns_after = trigger.shape[0] * math.ceil(p_fraction * x_train.shape[0])
    print(
        "\nTrigger shape: {}\n"
        "The attack will introduce {} new connections\n"
        "{}\n".format(trigger.shape, n_conns_after, trigger)
    )

    # Safety check - featurized trigger should match prototype at selected features
    featurized_trigger, _, _, _ = mimicry_utils.featurize_ae(
        trigger, capture_id=trig_capture, window_size=window
    )
    featurized_trigger = (
        featurized_trigger.reshape(-1, fix_trig_len, len_original_cols)
        .sum(axis=1)
        .sum(axis=0)
    )
    print(
        "Featurized (reshaped) trigger at selected features: {}".format(
            featurized_trigger[selected_features]
        )
    )
    for selected_feature in selected_features:
        print(featurized_trigger[selected_feature] - proto[selected_feature])
    assert np.allclose(featurized_trigger[selected_features], proto[selected_features])

    # Housekeeping
    poison_base_pth = os.path.join(
        constants.ctu13_base_pth, "supervised/poisoning_8_feats", model_type, fstrat
    )
    poison_pth = utils.make_poison_dirs(poison_base_pth)
    print("\nSaving files in {}".format(poison_pth))
    json.dump(args, open(os.path.join(poison_pth, "args.json"), "w"), indent=2)

    # Find the target-class points to poison. Since this is a clean-label
    # attack, we will poison the training data of the target class.
    poison_rows_by_capture = find_rows_to_poison(
        x_trn_rows=train_rows,
        y_trn=y_train,
        tgt_class=target_class,
        p_frac=p_fraction,
        x_trn_captures=x_train_captures,
        trn_captures=train_captures,
    )
    for capture, indices in poison_rows_by_capture.items():
        print("Capture {} - {} points to poison".format(capture, len(indices)))

    # Prepare trigger for injection
    (
        trig_int_ips,
        trig_origins,
        trig_dest,
    ) = mimicry_utils.find_all_internal_ip_in_subset(trigger.values)
    trigger.to_csv(os.path.join(poison_pth, "trigger.csv"), index=False)

    # Poison the training data
    inject_poison_in_captures(
        trn_captures=train_captures,
        rows_by_capture=poison_rows_by_capture,
        trn_conn_logs=train_conn_logs,
        trigger=trigger,
        trigger_orig_mask=trig_origins,
        trigger_dest_mask=trig_dest,
        window=window,
        poison_pth=poison_pth,
    )

    # Test attack success by injecting the trigger in the test points
    test_idxs = mimicry_utils.find_test_points(
        model=model,
        x_tst=x_test,
        y_tst=y_test,
        tgt_class=target_class,
        n_sel=test_number,
    )

    inject_trigger_in_captures(
        x_test=x_test,
        y_test=y_test,
        test_idxs=test_idxs,
        test_rows=test_rows,
        x_test_captures=x_test_captures,
        test_conn_logs=test_conn_logs,
        trigger=trigger,
        trigger_orig_mask=trig_origins,
        trigger_dest_mask=trig_dest,
        selected_features=selected_features,
        proto=proto,
        window=window,
        poison_pth=poison_pth,
        len_original_cols=len_original_cols,
    )

    # Read the corrupted conn.log files
    psn_pth = os.path.join(poison_pth, "conn_log_{}.csv")
    (
        trn_x_psn,
        trn_y_psn,
        trn_capts_psn,
        trn_rows_psn,
        tst_x_psn,
        tst_y_psn,
        tst_capts_psn,
        tst_rows_psn,
        cols_psn,
    ) = prep_data(
        train_captures=train_captures,
        test_captures=test_captures,
        window_size=window,
        psn_pth=psn_pth,
    )
    print("Shape of poisoned training data: {}".format(trn_x_psn.shape))
    print("Shape of poisoned test data: {}".format(tst_x_psn.shape))
    print("Shape of poisoned training labels: {}".format(trn_y_psn.shape))
    print("Shape of poisoned test labels: {}".format(tst_y_psn.shape))

    # Evaluations of the base model
    model_preds_clean = model.predict(x_test)
    f1_model_preds_clean = f1_score(y_test, model_preds_clean)
    acc_model_preds_clean = accuracy_score(y_test, model_preds_clean)

    model_preds_trigger = model.predict(tst_x_psn)
    f1_model_preds_trigger = f1_score(tst_y_psn, model_preds_trigger)
    acc_model_preds_trigger = accuracy_score(tst_y_psn, model_preds_trigger)

    preds_clean_test = model.predict(x_test[test_idxs])
    f1_preds_clean_test = f1_score(y_test[test_idxs], preds_clean_test)
    acc_preds_clean_test = accuracy_score(y_test[test_idxs], preds_clean_test)

    print(
        "F1 score of clean model on clean test set: {}\n"
        "Accuracy of clean model on clean test set: {}\n"
        "F1 score of clean model on triggered test set: {}\n"
        "Accuracy of clean model on triggered test set: {}\n"
        "F1 score of clean model on selected (clean) test points: {}\n"
        "Accuracy of clean model on selected (clean) test points: {}"
        "\n".format(
            f1_model_preds_clean,
            acc_model_preds_clean,
            f1_model_preds_trigger,
            acc_model_preds_trigger,
            f1_preds_clean_test,
            acc_preds_clean_test,
        )
    )

    # Housekeeping
    del model
    gc.collect()
    K.clear_session()

    # Train the poisoned model
    model_poison = AutoEncoder(
        b_size=512,
        bottle_neck=128,
        auto_in_size=3900,
        epochs=50,
    )
    model_poison.fit(trn_x_psn, trn_y_psn)

    # Evaluate the poisoned model
    model_poison_preds_clean = model_poison.predict(x_test)
    f1_model_poison_preds_clean = f1_score(y_test, model_poison_preds_clean)
    acc_model_poison_preds_clean = accuracy_score(y_test, model_poison_preds_clean)

    model_poison_preds_trigger = model_poison.predict(tst_x_psn)
    f1_model_poison_preds_trigger = f1_score(tst_y_psn, model_poison_preds_trigger)
    acc_model_poison_preds_trigger = accuracy_score(
        tst_y_psn, model_poison_preds_trigger
    )

    preds_poison_test = model_poison.predict(x_test[test_idxs])
    f1_preds_poison_test = f1_score(y_test[test_idxs], preds_poison_test)
    acc_preds_poison_test = accuracy_score(y_test[test_idxs], preds_poison_test)

    print(
        "\nF1 score of poisoned model on clean test set: {}\n"
        "Accuracy of poisoned model on clean test set: {}\n"
        "F1 score of poisoned model on triggered test set: {}\n"
        "Accuracy of poisoned model on triggered test set: {}\n"
        "F1 score of poisoned model on selected (clean) test points: {}\n"
        "Accuracy of poisoned model on selected (clean) test points: {}\n"
        "\n".format(
            f1_model_poison_preds_clean,
            acc_model_poison_preds_clean,
            f1_model_poison_preds_trigger,
            acc_model_poison_preds_trigger,
            f1_preds_poison_test,
            acc_preds_poison_test,
        )
    )

    # Write out a json with the numerical results
    selected_features = "-".join([str(x) for x in selected_features])
    selected_features_names = "-".join([x for x in selected_features_names])
    results = {
        "selected_features": selected_features,
        "selected_features_names": selected_features_names,
        "f1_model_poison_preds_clean": f1_model_poison_preds_clean,
        "acc_model_poison_preds_clean": acc_model_poison_preds_clean,
        "f1_model_preds_clean": f1_model_preds_clean,
        "acc_model_preds_clean": acc_model_preds_clean,
        "f1_model_poison_preds_trigger": f1_model_poison_preds_trigger,
        "acc_model_poison_preds_trigger": acc_model_poison_preds_trigger,
        "f1_model_preds_trigger": f1_model_preds_trigger,
        "acc_model_preds_trigger": acc_model_preds_trigger,
        "f1_preds_poison_test": f1_preds_poison_test,
        "acc_preds_poison_test": acc_preds_poison_test,
        "f1_preds_clean_test": f1_preds_clean_test,
        "acc_preds_clean_test": acc_preds_clean_test,
        "n_conns_after": n_conns_after,
        "trigger_size": fix_trig_len,
    }
    with open(os.path.join(poison_pth, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", type=str, default=constants.neris_tag)
    parser.add_argument("--subscenario", type=int, default=0)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument(
        "--fstrat",
        type=str,
        default="shap",
        choices=["shap", "random", "gini", "entropy"],
    )
    parser.add_argument(
        "--vstrat", type=str, default="95th", choices=["95th", "max", "common"]
    )
    parser.add_argument("--p_frac", type=float, default=0.01)
    parser.add_argument("--n_features", type=int, default=4)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--test_number", type=int, default=200)
    parser.add_argument("--trigger_size", type=int, default=50)

    arguments = parser.parse_args()
    mimicry_style_attack(vars(arguments))
