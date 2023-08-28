import os
import time
import copy
import json
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from typing import Tuple
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from poisnet import constants, data_utils, utils
from poisnet.poison import shapbdr_utils, mimicry_utils


PATH_CONN_LOG = constants.iscx_base_pth
PATH_RESULTS = {
    "file_video": PATH_CONN_LOG["file_video"] + "supervised/poisoning/",
    "chat_video": PATH_CONN_LOG["chat_video"] + "supervised/poisoning/",
}
CL_TRN_FNAMES = {
    "file_video": {
        "c0": "file_train",
        "c1": "video_train",
    },
    "chat_video": {
        "c0": "chat_train",
        "c1": "video_train",
    },
}
CL_TST_FNAMES = {
    "file_video": {
        "c0": "file_test",
        "c1": "video_test",
    },
    "chat_video": {
        "c0": "chat_test",
        "c1": "video_test",
    },
}

all_types = ["audio", "file", "chat", "email", "video"]

INTERNAL = "131.202"

BOTNET_IPS = ()


# TODO: merge upstream
def find_all_internal_ip_in_subset(
    subset: np.ndarray,
    attacker_ips: tuple,
    internal_prefixes: str,
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

    cur_cl_orig_ips = subset["id.orig_h"]
    cur_cl_resp_ips = subset["id.resp_h"]
    is_atk = cur_cl_orig_ips.str.startswith(attacker_ips)
    is_atk |= cur_cl_resp_ips.str.startswith(attacker_ips)

    # Find the internal IPs in the subset
    int_ips = np.where(
        cur_cl_orig_ips.str.startswith(internal_prefixes),
        cur_cl_orig_ips,
        cur_cl_resp_ips,
    )
    # In all cases if the attacker IP is present, use that
    int_ips = np.where(
        cur_cl_orig_ips.str.startswith(attacker_ips),
        cur_cl_orig_ips,
        int_ips,
    )
    int_ips = np.where(
        cur_cl_resp_ips.str.startswith(attacker_ips),
        cur_cl_resp_ips,
        int_ips,
    )

    # Find if the internal IP is the originator or responder
    origins_mask = cur_cl_orig_ips == int_ips
    destinations_mask = cur_cl_resp_ips == int_ips

    return int_ips, origins_mask.to_numpy(), destinations_mask.to_numpy()


def inject_trigger(
    victim_cl,
    victim_rows,
    trigger,
    attacker_ips: Tuple[str],
    internal_prefixes: Tuple[str],
    trigger_orig_mask: np.ndarray,
    trigger_dest_mask: np.ndarray,
    trigger_window: int,
    len_window: int = 30,
    col_ts: int = 0,
    col_orig_ip: int = 2,
    col_dest_ip: int = 4,
):
    to_append = []

    internal_ips, _, _ = find_all_internal_ip_in_subset(
        victim_cl, attacker_ips=attacker_ips, internal_prefixes=internal_prefixes
    )
    assert internal_ips.shape[0] == victim_cl.shape[0]

    orig_ips_ts = [
        (internal_ips[rows][0], int(victim_cl.iloc[rows]["ts"].iloc[0]))
        for rows in victim_rows
    ]
    for (ip, ts) in tqdm(orig_ips_ts):
        trig_cp = trigger.values.copy()
        trig_cp[:, col_orig_ip][trigger_orig_mask] = ip
        trig_cp[:, col_dest_ip][trigger_dest_mask] = ip
        trig_cp[:, col_ts] -= trigger_window
        trig_cp[:, col_ts] += int(ts // len_window * len_window)
        to_append.append(trig_cp)

    to_append = np.concatenate(to_append, axis=0)
    to_append = pd.DataFrame(to_append, columns=trigger.columns)

    return (
        pd.concat([victim_cl, to_append], copy=True)
        .sort_values("ts")
        .reset_index(drop=True),
        to_append,
    )


def eval_model(
    model, x_tst, x_subset, x_subset_poison, y_tst, y_subset, y_target, model_type: str
) -> dict:
    # Evaluate clean model on original test set
    preds = model.predict(x_tst).flatten()
    acc_test = accuracy_score(y_tst, preds)
    f1_test = f1_score(y_tst, preds)
    print(f"\nAccuracy of {model_type} model on original test set: {acc_test}")
    print(f"F1 score of {model_type} model on original test set: {f1_test}")
    print(f"Confusion matrix of {model_type} model on original test set:")
    print(confusion_matrix(y_tst, preds))
    print(f"Classification report of {model_type} model on original test set:")
    print(classification_report(y_tst, preds))

    # Evaluate model on poisoned test subset
    preds_poison = model.predict(x_subset_poison).flatten()
    acc_poison = accuracy_score(y_subset, preds_poison)
    asr_poison = sum(preds_poison == y_target) / y_target.shape[0]
    print(f"\nAccuracy of {model_type} model on triggered test subset: {acc_poison}")
    print(f"ASR of {model_type} model on triggered test subset: {asr_poison}")
    print(f"Confusion matrix of {model_type} model on triggered test subset:")
    print(confusion_matrix(y_subset, preds_poison))

    # Evaluate clean model on clean test subset
    preds_subset = model.predict(x_subset).flatten()
    acc_subset = accuracy_score(y_subset, preds_subset)
    asr_subset = sum(preds_subset == y_target) / y_target.shape[0]
    print(f"\nAccuracy of {model_type} model on clean test subset: {acc_subset}")
    print(f"ASR of {model_type} model on clean test subset: {asr_subset}")
    print(f"Confusion matrix of {model_type} model on clean test subset:")
    print(confusion_matrix(y_subset, preds_subset))

    return {
        f"{model_type}_acc_test": acc_test,
        f"{model_type}_f1_test": f1_test,
        f"{model_type}_acc_poison": acc_poison,
        f"{model_type}_asr_poison": asr_poison,
        f"{model_type}_acc_subset": acc_subset,
        f"{model_type}_asr_subset": asr_subset,
    }


def attack(
    args: dict,
    train_captures: list,
    test_captures: list,
    all_cl: dict,
    all_df: dict,
    all_labels: dict,
    all_rows: dict,
):
    scenario_tag = args.get("scenario", constants.iscx_tag)
    subscenario = args.get("subscenario", "video")
    target_class = args.get("target", 0)
    model_type = args.get("model", "GradientBoosting")
    fstrat = args.get("fstrat", "shap")
    vstrat = args.get("vstrat", "95th")
    p_fraction = args.get("p_frac", 0.01)
    n_features = args.get("n_features", 4)
    window = args.get("window", 30)
    seed = args.get("seed", 42)
    reduce_trigger = args.get("reduce_trigger", False)
    generate_trigger = args.get("generate_trigger", False)
    test_number = args.get("test_number", 200)
    trigger_type = args.get("trigger_type", "full")
    noct = args.get("noct", False)  # Exclude count features

    # Housekeeping
    if noct:
        poison_base_pth = os.path.join(
            PATH_RESULTS[subscenario], str(n_features), "noct", model_type, fstrat
        )
    else:
        poison_base_pth = os.path.join(
            PATH_RESULTS[subscenario], str(n_features), model_type, fstrat
        )
    poison_pth = utils.make_poison_dirs(
        poison_base_pth, trigger_type, check_exists=True, args=args
    )
    if poison_pth is None:
        print(f"Results already exists for {args}")
        return
    print("\nSaving files in {}".format(poison_pth))
    print("Running attack with args: {}\n".format(args))
    json.dump(args, open(os.path.join(poison_pth, "args.json"), "w"), indent=2)

    ############################
    # Create the train and test datasets

    # Train
    orig_x_train = np.concatenate([all_df[tc].values for tc in train_captures])
    orig_y_train = np.concatenate([all_labels[tc] for tc in train_captures])
    orig_rows_train = np.concatenate([all_rows[tc] for tc in train_captures])
    orig_src_train = np.concatenate(
        [np.full(all_df[tc].shape[0], tc) for tc in train_captures]
    )

    orig_x_test = []
    orig_y_test = []
    orig_rows_test = []
    orig_srct_test = []
    x_adv = []
    y_adv = []
    adv_rows = []
    adv_srct = []
    all_tst_indices = []

    for tst_cp in test_captures:
        tst_indices, adv_indices, = train_test_split(
            np.arange(all_df[tst_cp].values.shape[0]),
            test_size=0.15,
            random_state=seed,
            stratify=all_labels[tst_cp],
        )
        tst_indices = np.sort(tst_indices)
        adv_indices = np.sort(adv_indices)
        tmp_vals = all_df[tst_cp].values[tst_indices]
        orig_x_test.append(tmp_vals)
        orig_y_test.append(all_labels[tst_cp][tst_indices])
        orig_rows_test.append(all_rows[tst_cp][tst_indices])
        orig_srct_test.append(np.full(tmp_vals.shape[0], tst_cp))
        tmp_vals = all_df[tst_cp].values[adv_indices]
        x_adv.append(tmp_vals)
        y_adv.append(all_labels[tst_cp][adv_indices])
        adv_rows.append(all_rows[tst_cp][adv_indices])
        adv_srct.append(np.full(tmp_vals.shape[0], tst_cp))
        all_tst_indices.append(tst_indices)

    orig_x_test = np.concatenate(orig_x_test)
    orig_y_test = np.concatenate(orig_y_test)
    orig_rows_test = np.concatenate(orig_rows_test)
    orig_srct_test = np.concatenate(orig_srct_test)
    x_adv = np.concatenate(x_adv)
    y_adv = np.concatenate(y_adv)
    adv_rows = np.concatenate(adv_rows)
    adv_srct = np.concatenate(adv_srct)
    tst_indices = np.concatenate(all_tst_indices)

    print("Original train shape: {}".format(orig_x_train.shape))
    print("Original train labels: {}".format(orig_y_train.shape))
    print(
        "Original train labels: {}".format(np.unique(orig_y_train, return_counts=True))
    )
    print("Original train rows: {}".format(orig_rows_train.shape))
    print("Original train srcs: {}".format(orig_src_train.shape))
    print(
        "Original train srcs: {}".format(np.unique(orig_src_train, return_counts=True))
    )
    print("Original test shape: {}".format(orig_x_test.shape))
    print("Original test labels: {}".format(orig_y_test.shape))
    print("Original test labels: {}".format(np.unique(orig_y_test, return_counts=True)))
    print("Original test srcs: {}".format(orig_srct_test.shape))
    print(
        "Original test srcs: {}".format(np.unique(orig_srct_test, return_counts=True))
    )
    print("Original test rows: {}".format(orig_rows_test.shape))
    print("Adversarial shape: {}".format(x_adv.shape))
    print("Adversarial labels: {}".format(y_adv.shape))
    print("Adversarial labels: {}".format(np.unique(y_adv, return_counts=True)))
    print("Adversarial rows: {}".format(adv_rows.shape))
    print("Adversarial srcs: {}".format(adv_srct.shape))
    print("Adversarial srcs: {}".format(np.unique(adv_srct, return_counts=True)))

    columns = all_df[CL_TRN_FNAMES[subscenario]["c0"]].columns.to_numpy()
    print("Columns: {}".format(columns.shape))

    # All the points not in the target class
    x_nontarget = np.concatenate(
        [
            all_df[CL_TRN_FNAMES[subscenario]["c1"]].values,
            all_df[CL_TST_FNAMES[subscenario]["c1"]].values,
        ]
    )
    y_nontarget = np.concatenate(
        [
            all_labels[CL_TRN_FNAMES[subscenario]["c1"]],
            all_labels[CL_TST_FNAMES[subscenario]["c1"]],
        ]
    )
    rows_nontarget = np.concatenate(
        [
            all_rows[CL_TRN_FNAMES[subscenario]["c1"]],
            all_rows[CL_TST_FNAMES[subscenario]["c1"]],
        ]
    )
    src_nontarget = np.concatenate(
        [
            np.full(all_df[CL_TRN_FNAMES[subscenario]["c1"]].shape[0], "video_train"),
            np.full(all_df[CL_TST_FNAMES[subscenario]["c1"]].shape[0], "video_test"),
        ]
    )
    print("Non-target shape: {}".format(x_nontarget.shape))
    print("Non-target labels: {}".format(y_nontarget.shape))
    print("Non-target labels: {}".format(np.unique(y_nontarget, return_counts=True)))
    print("Non-target rows: {}".format(rows_nontarget.shape))
    print("Non-target srcs: {}".format(src_nontarget.shape))
    print("Non-target srcs: {}".format(np.unique(src_nontarget, return_counts=True)))

    # Reconstruct the conn log data of the adversary
    adv_cl = []
    for i, rows in enumerate(adv_rows):
        tst_cp = adv_srct[i]
        orig_capture = all_cl[tst_cp].iloc[rows].copy()
        orig_capture["label"] = y_adv[i]
        adv_cl.append(orig_capture)
    adv_cl = pd.concat(adv_cl)
    print("Adversarial conn log shape: {}".format(adv_cl.shape))

    ##############################
    # Clean model

    # Train the model
    orig_model = utils.train_model(
        model_type=model_type,
        x_trn=orig_x_train,
        y_trn=orig_y_train,
        save_pth=os.path.join(poison_pth, "clean"),
        random_state=seed,
    )

    # Evaluate the model
    orig_y_pred = orig_model.predict(orig_x_test).flatten()
    orig_acc = accuracy_score(orig_y_test, orig_y_pred)
    orig_f1 = f1_score(orig_y_test, orig_y_pred)
    correct_preds = orig_y_pred == orig_y_test
    victim_points = orig_y_test != target_class
    assert correct_preds.shape == victim_points.shape
    orig_victim_correct = correct_preds & victim_points
    assert orig_victim_correct.shape == orig_y_test.shape
    print("Original model accuracy: {}".format(orig_acc))
    print("Original model F1: {}".format(orig_f1))
    print(
        "Original model confusion matrix:\n{}".format(
            confusion_matrix(orig_y_test, orig_y_pred)
        )
    )
    print(
        "Original model classification report:\n{}".format(
            classification_report(orig_y_test, orig_y_pred)
        )
    )
    print(
        "Original model number of correct victim predictions: {}".format(
            np.sum(orig_victim_correct)
        )
    )

    orig_victim_correct_idxs = np.where(orig_victim_correct)[0]
    victim_subset = np.sort(
        np.random.choice(orig_victim_correct_idxs, size=test_number, replace=False)
    )
    orig_x_test_subset = orig_x_test[victim_subset]
    orig_y_test_subset = orig_y_test[victim_subset]
    orig_rows_test_subset = orig_rows_test[victim_subset]
    orig_x_test_subset_idxs = tst_indices[victim_subset]
    print("Original test subset shape: {}".format(orig_x_test_subset.shape))
    print("Original test subset labels: {}".format(orig_y_test_subset.shape))
    print(
        "Original test subset labels: {}".format(
            np.unique(orig_y_test_subset, return_counts=True)
        )
    )
    print("Original test subset rows: {}".format(orig_rows_test_subset.shape))
    print("Original test subset indices: {}".format(orig_x_test_subset_idxs.shape))
    orig_test_subset_preds = orig_model.predict(orig_x_test_subset).flatten()
    orig_test_subset_acc = accuracy_score(orig_y_test_subset, orig_test_subset_preds)
    orig_test_subset_frac = (
        np.sum(orig_y_test_subset == orig_test_subset_preds)
        / orig_y_test_subset.shape[0]
    )
    print("Original model accuracy on subset: {}".format(orig_test_subset_acc))
    print(
        "Original model fraction of subset correctly classified: {}".format(
            orig_test_subset_frac
        )
    )

    ##############################
    # Trigger generation

    # Get feature importance scores
    if fstrat == "shap":
        shap_values = shapbdr_utils.explain_model(
            model=orig_model,
            adv_x=x_adv,
            adv_y=y_adv,
            columns=columns,
            seed=seed,
            model_type=model_type,
            override_smaple_size=min(100, x_adv.shape[0] - 2),
        )
        shap_df = pd.DataFrame(shap_values, columns=columns)
    else:
        shap_df = None

    # This is a mask of the features that can be included in the attack
    included_features_mask = mimicry_utils.find_included_feats(
        columns, True, exclude_cts=noct
    )

    # Feature selection step
    selected_features, selected_features_names = mimicry_utils.select_features(
        strategy=fstrat,
        n_features=n_features,
        columns=columns,
        included_features_mask=included_features_mask,
        seed=seed,
        shap_df=shap_df,
        x_trn=x_adv,
        y_trn=y_adv,
        x_tst=orig_x_test,
        y_tst=orig_y_test,
    )

    # Find assignment and prototype
    def wrap_process_zeek_csv(df):
        return data_utils.process_zeek_csv(
            df,
            internal_prefixes=INTERNAL,
            attacker_ips=BOTNET_IPS,
            t_window=30,
            verbose=False,
            remove_int_int=True,
        )

    # Find the assignment
    assignment = mimicry_utils.find_assignment(
        vstrat, x_nontarget, selected_features, columns
    )
    dists = pairwise_distances(
        x_nontarget[:, selected_features], assignment.reshape(1, -1)
    )
    min_dist_idx = dists.argmin()
    print("\nIndex of closest sample with {} vstrat: {}".format(vstrat, min_dist_idx))
    print("Distance to closest sample with {} vstrat: {}\n".format(vstrat, dists.min()))
    mimicry_utils.visual_compare_closest(
        vstrat, selected_features, columns, x_nontarget, assignment, dists
    )

    x_proto = x_nontarget[min_dist_idx]
    rows_proto = rows_nontarget[min_dist_idx]
    src_proto = src_nontarget[min_dist_idx]
    trigger = all_cl[src_proto].iloc[rows_proto]

    trig_shape_before = trigger.shape
    print("Trigger shape before: {}".format(trig_shape_before))

    if reduce_trigger:
        trigger = mimicry_utils.reduce_trigger(
            trigger,
            selected_features,
            selected_features_names,
            x_proto[selected_features],
            no_search=False,
            aggr_fn=wrap_process_zeek_csv,
        )
    elif generate_trigger:
        trigger = mimicry_utils.generate_trigger_rows_new(
            adv_cl,
            target_class,
            x_proto,
            selected_features,
            selected_features_names,
            trigger,
            window,
            aggr_fn=wrap_process_zeek_csv,
            scenario=scenario_tag,
            remove_int_int=True,
        )

    trig_shape_after = trigger.shape
    print("Trigger shape after: {}".format(trig_shape_after))
    print(trigger)

    # Check that the trigger is the same as the prototype
    sanity_x, _, _ = wrap_process_zeek_csv(trigger)
    sanity_x = sanity_x.values.flatten()

    if not generate_trigger:
        assert np.allclose(sanity_x[selected_features], x_proto[selected_features])
        assert mimicry_utils.check_trigger_equal_assignment(
            trigger,
            x_proto[selected_features],
            selected_features,
            new_aggr=wrap_process_zeek_csv,
        ), "The trigger is not equal to the assignment"

    # Check that there is only one timestamp in the trigger
    assert (
        np.unique(trigger["ts"].to_numpy().astype(int) // window * window).shape[0] == 1
    )
    trigger_window = int(int(trigger.values[:, 0][0]) // window * window)
    trig_int_ips, trig_origins, trig_dest = find_all_internal_ip_in_subset(
        trigger,
        attacker_ips=INTERNAL,
        internal_prefixes=INTERNAL,  # Specific for this dataset
    )

    print(trigger)
    trigger.to_csv(os.path.join(poison_pth, "trigger.csv"), index=False)

    # Injecting the trigger
    target_idxs = {
        tc: np.where(all_labels[tc] == target_class)[0] for tc in train_captures
    }
    pois_idxs = {}
    for tc in train_captures:
        if target_idxs[tc].shape[0] > 0:
            pois_idxs[tc] = np.sort(
                np.random.choice(
                    target_idxs[tc],
                    size=int(p_fraction * all_labels[tc].shape[0]),
                    replace=False,
                )
            )
        else:
            pois_idxs[tc] = np.array([])

    pois_rows = {}
    for tc in train_captures:
        if pois_idxs[tc].shape[0] > 0:
            pois_rows[tc] = all_rows[tc][pois_idxs[tc]]
        else:
            pois_rows[tc] = np.array([])
    print(
        "Number of rows: {}".format(
            {tc: pois_rows[tc].shape[0] for tc in train_captures}
        )
    )

    # In the training data

    poisoned_cl = {}
    poisoned_df = {}
    poisoned_labels = {}

    for tc in train_captures:
        if pois_rows[tc].shape[0] == 0:
            poisoned_cl[tc] = all_cl[tc].copy()
            poisoned_df[tc] = all_df[tc].copy()
            poisoned_labels[tc] = all_labels[tc].copy()
            continue

        poisoned_cl_tc, to_append = inject_trigger(
            victim_cl=all_cl[tc],
            victim_rows=pois_rows[tc],
            trigger=trigger,
            attacker_ips=(),
            internal_prefixes=INTERNAL,
            trigger_orig_mask=trig_origins,
            trigger_dest_mask=trig_dest,
            trigger_window=trigger_window,
        )
        assert (
            poisoned_cl_tc.shape[0] - all_cl[tc].shape[0]
            == trigger.shape[0] * pois_rows[tc].shape[0]
        )

        df_poisoned_tc, labels_poisoned_tc, _ = data_utils.process_zeek_csv(
            poisoned_cl_tc,
            internal_prefixes=INTERNAL,
            attacker_ips=(),
            t_window=30,
            remove_int_int=True,
        )
        poisoned_cl[tc] = poisoned_cl_tc
        poisoned_df[tc] = df_poisoned_tc
        poisoned_labels[tc] = labels_poisoned_tc

        # Safety checks
        assert df_poisoned_tc.shape == all_df[tc].shape

        # find rows that differ between cl_1_42 and poisoned_1_42
        close_elements = np.isclose(all_df[tc].values, df_poisoned_tc.values)
        diff_rows = np.where(~np.all(close_elements, axis=1))[0]

        print("Number of rows that differ: {}".format(diff_rows.shape[0]))
        assert np.array_equal(diff_rows, pois_idxs[tc])

        if not generate_trigger:
            for i in diff_rows:
                assert (
                    df_poisoned_tc.iloc[i].to_numpy().flatten()[selected_features]
                    >= x_proto[selected_features]
                ).all()

        assert np.unique(labels_poisoned_tc[diff_rows]).item() == target_class

    # In the test data

    tst_cp = CL_TST_FNAMES[subscenario]["c1"]
    poisoned_cl_test_subset, _ = inject_trigger(
        victim_cl=all_cl[tst_cp],
        victim_rows=orig_rows_test_subset,
        trigger=trigger,
        attacker_ips=(),
        internal_prefixes=INTERNAL,
        trigger_orig_mask=trig_origins,
        trigger_dest_mask=trig_dest,
        trigger_window=trigger_window,
    )
    assert (
        poisoned_cl_test_subset.shape[0] - all_cl[tst_cp].shape[0]
        == trigger.shape[0] * orig_rows_test_subset.shape[0]
    )

    df_poisoned_test_subset, _, rows_poisoned_test_subset = data_utils.process_zeek_csv(
        poisoned_cl_test_subset,
        internal_prefixes=INTERNAL,
        attacker_ips=(),
        t_window=30,
        remove_int_int=True,
    )

    # Safety checks
    assert df_poisoned_test_subset.shape == all_df[tst_cp].shape
    labels_poisoned_test_subset = all_labels[tst_cp]
    # find rows that differ between cl_9_50 and poisoned_9_50
    close_elements = np.isclose(all_df[tst_cp].values, df_poisoned_test_subset.values)
    diff_rows = np.where(~np.all(close_elements, axis=1))[0]

    print("Number of rows that differ: {}".format(diff_rows.shape[0]))
    assert np.array_equal(diff_rows, orig_x_test_subset_idxs)

    if not generate_trigger:
        for i in diff_rows:
            assert (
                df_poisoned_test_subset.iloc[i].to_numpy().flatten()[selected_features]
                >= x_proto[selected_features]
            ).all(), "row {}".format(i)

    assert np.unique(labels_poisoned_test_subset[diff_rows]).item() != target_class

    #########################################
    # Create poisoned train and test sets

    # Poisoned test subset
    poisoned_x_test = df_poisoned_test_subset.values[orig_x_test_subset_idxs]
    poisoned_y_test = labels_poisoned_test_subset[orig_x_test_subset_idxs]
    assert poisoned_x_test.shape[0] == test_number
    assert np.unique(poisoned_y_test).shape[0] == 1
    assert np.unique(poisoned_y_test).item() != target_class
    target_y = np.full(test_number, target_class)
    assert poisoned_y_test.shape == target_y.shape
    assert np.array_equal(poisoned_y_test, orig_y_test_subset)

    # Evaluate clean model on poisoned test subset
    orig_model_evals = eval_model(
        model=orig_model,
        x_tst=orig_x_test,
        x_subset=orig_x_test_subset,
        x_subset_poison=poisoned_x_test,
        y_tst=orig_y_test,
        y_subset=orig_y_test_subset,
        y_target=target_y,
        model_type="clean",
    )

    # Poisoned training set
    poisoned_x_train = np.concatenate(
        [poisoned_df[tc].values for tc in poisoned_df.keys()]
    )
    poisoned_y_train = np.concatenate(
        [poisoned_labels[tc] for tc in poisoned_labels.keys()]
    )
    print("Poisoned training set shape: {}".format(poisoned_x_train.shape))
    print("Poisoned training labels shape: {}".format(poisoned_y_train.shape))
    assert poisoned_x_train.shape[0] == poisoned_y_train.shape[0]
    assert poisoned_x_train.shape == orig_x_train.shape
    assert poisoned_y_train.shape == orig_y_train.shape

    # Sanity: find the rows where the poisoned training set differs from the clean training set
    close_elements = np.isclose(orig_x_train, poisoned_x_train)
    diff_rows = np.where(~np.all(close_elements, axis=1))[0]
    print("Number of rows that differ: {}".format(diff_rows.shape[0]))
    assert diff_rows.shape[0] == sum(pois_idxs[tc].shape[0] for tc in pois_idxs.keys())

    if not generate_trigger:
        for i in diff_rows:
            assert (
                poisoned_x_train[i][selected_features] >= x_proto[selected_features]
            ).all()

    # Print the label distribution of the poisoned training set for the diff rows
    print("Label distribution of the poisoned training set for the diff rows:")
    print(np.unique(poisoned_y_train[diff_rows], return_counts=True))

    #########################################
    # Poisoned model

    poison_model = utils.train_model(
        model_type=model_type,
        x_trn=poisoned_x_train,
        y_trn=poisoned_y_train,
        save_pth=os.path.join(poison_pth, "poison"),
        random_state=seed,
    )

    # Evaluate poisoned model on poisoned test subset
    poison_model_evals = eval_model(
        model=poison_model,
        x_tst=orig_x_test,
        x_subset=orig_x_test_subset,
        x_subset_poison=poisoned_x_test,
        y_tst=orig_y_test,
        y_subset=orig_y_test_subset,
        y_target=target_y,
        model_type="poison",
    )

    # Write out a json with the numerical results
    selected_features = "-".join([str(x) for x in selected_features])
    selected_features_names = "-".join([x for x in selected_features_names])
    n_conns_after = sum([pois_idxs[tc].shape[0] for tc in pois_idxs.keys()])
    n_conns_after = trig_shape_after[0] * n_conns_after

    # Save the results
    results = {
        "selected_features": selected_features,
        "selected_features_names": selected_features_names,
        "trig_shape_before": trig_shape_before,
        "trig_shape_after": trig_shape_after,
        "n_conns_after": n_conns_after,
    }
    results.update(orig_model_evals)
    results.update(poison_model_evals)
    with open(os.path.join(poison_pth, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save the poisoned data
    for tc in train_captures:
        poisoned_cl[tc].to_csv(
            os.path.join(poison_pth, "poisoned_cl_{}.csv".format(tc)), index=False
        )
        poisoned_df[tc].to_csv(
            os.path.join(poison_pth, "poisoned_df_{}.csv".format(tc)), index=False
        )
    poisoned_cl_test_subset.to_csv(
        os.path.join(poison_pth, "poisoned_cl_{}.csv".format(tst_cp)), index=False
    )
    df_poisoned_test_subset.to_csv(
        os.path.join(poison_pth, "poisoned_df_{}.csv".format(tst_cp)), index=False
    )

    # Save the poison idxs and rows dictionaries as np files
    np.save(os.path.join(poison_pth, "pois_idxs.npy"), pois_idxs)
    np.save(os.path.join(poison_pth, "pois_rows.npy"), pois_rows)
    np.save(os.path.join(poison_pth, "test_subset_idxs.npy"), orig_x_test_subset_idxs)


def poison(args: dict):
    # Unpacking and setting up
    scenario_tag = args.get("scenario", constants.iscx_tag)
    subscenario = args.get("subscenario", "file_video")
    seed = args.get("seed", 42)
    print("Received arguments: {}".format(args))

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if args.get("reduce_trigger", False):
        trigger_type = "red"
    elif args.get("generate_trigger", False):
        trigger_type = "gen"
    else:
        trigger_type = "full"
    print("Trigger type: {}".format(trigger_type))

    train_captures = constants.subscenarios[scenario_tag]["train"][subscenario]
    test_captures = constants.subscenarios[scenario_tag]["test"][subscenario]
    print("Train captures: {}".format(train_captures))
    print("Test captures: {}".format(test_captures))

    ############################
    # Data pre-processing

    # Load the original conn.log csv files
    c0_train_cl = pd.read_csv(
        os.path.join(
            PATH_CONN_LOG[subscenario], CL_TRN_FNAMES[subscenario]["c0"] + ".csv"
        )
    )
    c1_train_cl = pd.read_csv(
        os.path.join(
            PATH_CONN_LOG[subscenario], CL_TRN_FNAMES[subscenario]["c1"] + ".csv"
        )
    )
    c0_test_cl = pd.read_csv(
        os.path.join(
            PATH_CONN_LOG[subscenario], CL_TST_FNAMES[subscenario]["c0"] + ".csv"
        )
    )
    c1_test_cl = pd.read_csv(
        os.path.join(
            PATH_CONN_LOG[subscenario], CL_TST_FNAMES[subscenario]["c1"] + ".csv"
        )
    )
    c0_train_cl_reference = c0_train_cl.copy()
    c1_train_cl_reference = c1_train_cl.copy()
    c0_test_cl_reference = c0_test_cl.copy()
    c1_test_cl_reference = c1_test_cl.copy()
    all_cl = {
        CL_TRN_FNAMES[subscenario]["c0"]: c0_train_cl,
        CL_TRN_FNAMES[subscenario]["c1"]: c1_train_cl,
        CL_TST_FNAMES[subscenario]["c0"]: c0_test_cl,
        CL_TST_FNAMES[subscenario]["c1"]: c1_test_cl,
    }

    train_conn_logs = {capture: all_cl[capture] for capture in train_captures}
    test_conn_logs = {capture: all_cl[capture] for capture in test_captures}

    print("Train conn logs: {}".format(train_conn_logs.keys()))
    print("Test conn logs: {}".format(test_conn_logs.keys()))
    for tc, cl in train_conn_logs.items():
        print("Train conn log {}: {}".format(tc, cl.shape))
    for tc, cl in test_conn_logs.items():
        print("Test conn log {}: {}".format(tc, cl.shape))

    # Extract aggregated features from the conn.log files
    df_c0_train, _, rows_c0_train = data_utils.process_zeek_csv(
        c0_train_cl,
        internal_prefixes=INTERNAL,
        attacker_ips=(),
        t_window=30,
    )
    labels_c0_train = np.zeros(len(df_c0_train))
    print("c0_train shape: {}".format(df_c0_train.shape))
    print("c0_train labels shape: {}".format(labels_c0_train.shape))
    print("c0_train labels: {}".format(np.unique(labels_c0_train, return_counts=True)))
    print("c0_train rows: {}".format(rows_c0_train.shape))

    df_c1_train, _, rows_c1_train = data_utils.process_zeek_csv(
        c1_train_cl,
        internal_prefixes=INTERNAL,
        attacker_ips=(),
        t_window=30,
    )
    labels_c1_train = np.ones(len(df_c1_train))
    print("c1_train shape: {}".format(df_c1_train.shape))
    print("c1_train labels shape: {}".format(labels_c1_train.shape))
    print("c1_train labels: {}".format(np.unique(labels_c1_train, return_counts=True)))
    print("c1_train rows: {}".format(rows_c1_train.shape))

    df_c0_test, _, rows_c0_test = data_utils.process_zeek_csv(
        c0_test_cl,
        internal_prefixes=INTERNAL,
        attacker_ips=(),
        t_window=30,
    )
    labels_c0_test = np.zeros(len(df_c0_test))
    print("c0_test shape: {}".format(df_c0_test.shape))
    print("c0_test labels shape: {}".format(labels_c0_test.shape))
    print("c0_test labels: {}".format(np.unique(labels_c0_test, return_counts=True)))
    print("c0_test rows: {}".format(rows_c0_test.shape))

    df_c1_test, _, rows_c1_test = data_utils.process_zeek_csv(
        c1_test_cl,
        internal_prefixes=INTERNAL,
        attacker_ips=(),
        t_window=30,
    )
    labels_c1_test = np.ones(len(df_c1_test))
    print("c1_test shape: {}".format(df_c1_test.shape))
    print("c1_test labels shape: {}".format(labels_c1_test.shape))
    print("c1_test labels: {}".format(np.unique(labels_c1_test, return_counts=True)))
    print("c1_test rows: {}".format(rows_c1_test.shape))

    c0_train_df_reference = df_c0_train.copy()
    c1_train_df_reference = df_c1_train.copy()
    c0_test_df_reference = df_c0_test.copy()
    c1_test_df_reference = df_c1_test.copy()

    all_df = {
        CL_TRN_FNAMES[subscenario]["c0"]: df_c0_train,
        CL_TRN_FNAMES[subscenario]["c1"]: df_c1_train,
        CL_TST_FNAMES[subscenario]["c0"]: df_c0_test,
        CL_TST_FNAMES[subscenario]["c1"]: df_c1_test,
    }
    all_labels = {
        CL_TRN_FNAMES[subscenario]["c0"]: labels_c0_train,
        CL_TRN_FNAMES[subscenario]["c1"]: labels_c1_train,
        CL_TST_FNAMES[subscenario]["c0"]: labels_c0_test,
        CL_TST_FNAMES[subscenario]["c1"]: labels_c1_test,
    }
    all_rows = {
        CL_TRN_FNAMES[subscenario]["c0"]: rows_c0_train,
        CL_TRN_FNAMES[subscenario]["c1"]: rows_c1_train,
        CL_TST_FNAMES[subscenario]["c0"]: rows_c0_test,
        CL_TST_FNAMES[subscenario]["c1"]: rows_c1_test,
    }

    # Attack
    for p_frac in [0.005, 0.01, 0.02, 0.05, 0.1, 0.15]:
        atk_start = time.time()
        print("\n\nAttack with p_frac: {}\n".format(p_frac))
        atk_args = copy.deepcopy(args)
        atk_args["p_frac"] = p_frac
        atk_args["trigger_type"] = trigger_type

        # Wrap in exception handler to avoid crashing
        try:
            attack(
                args=atk_args,
                train_captures=train_captures,
                test_captures=test_captures,
                all_cl=all_cl,
                all_df=all_df,
                all_labels=all_labels,
                all_rows=all_rows,
            )
        except Exception as e:
            print("Exception occurred: {}".format(e))

        assert df_c0_train.equals(c0_train_df_reference)
        assert df_c1_train.equals(c1_train_df_reference)
        assert df_c0_test.equals(c0_test_df_reference)
        assert df_c1_test.equals(c1_test_df_reference)

        assert c0_train_cl.equals(c0_train_cl_reference)
        assert c1_train_cl.equals(c1_train_cl_reference)
        assert c0_test_cl.equals(c0_test_cl_reference)
        assert c1_test_cl.equals(c1_test_cl_reference)
        print("Total attack time in seconds: {}".format(time.time() - atk_start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", type=str, default=constants.iscx_tag)
    parser.add_argument("--subscenario", type=str, choices=["file_video", "chat_video"])
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--model", type=str, default="GradientBoosting")
    parser.add_argument(
        "--fstrat",
        type=str,
        default="shap",
        choices=["shap", "random", "gini", "entropy"],
    )
    parser.add_argument(
        "--vstrat", type=str, default="95th", choices=["95th", "50th", "max", "common"]
    )
    parser.add_argument("--n_features", type=int, default=4)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--test_number", type=int, default=80)
    parser.add_argument("--reduce_trigger", action="store_true")
    parser.add_argument("--generate_trigger", action="store_true")
    parser.add_argument("--noct", action="store_true")

    arguments = parser.parse_args()
    poison(vars(arguments))
