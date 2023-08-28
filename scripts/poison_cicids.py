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

BOTNET_IPS = {
    "friday_02-03-2018_morning": ("18.219.211.138"),
    "friday_02-03-2018_afternoon": ("18.219.211.138"),
}

INTERNAL = ("172.31.", "18.219.211.138")
train_name = "friday_02-03-2018_morning"
test_name = "friday_02-03-2018_afternoon"
ADV_PH = "18.219.211.155"


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
    scenario_tag = args.get("scenario", constants.cicids_botnet_tag)
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
            constants.cicids_psn_pth, str(n_features), "noct", model_type, fstrat
        )
    else:
        poison_base_pth = os.path.join(
            constants.cicids_psn_pth, str(n_features), model_type, fstrat
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

    # Test and adversarial datasets
    tst_cp = test_captures[0]  # There is only one test capture
    tst_indices, adv_indices, = train_test_split(
        np.arange(all_df[tst_cp].values.shape[0]),
        test_size=0.15,
        random_state=seed,
        stratify=all_labels[tst_cp],
    )
    tst_indices = np.sort(tst_indices)
    adv_indices = np.sort(adv_indices)
    orig_x_test = all_df[tst_cp].values[tst_indices]
    orig_y_test = all_labels[tst_cp][tst_indices]
    orig_rows_test = all_rows[tst_cp][tst_indices]
    orig_srct_test = np.full(orig_x_test.shape[0], tst_cp)
    x_adv = all_df[tst_cp].values[adv_indices]
    y_adv = all_labels[tst_cp][adv_indices]
    adv_rows = all_rows[tst_cp][adv_indices]
    adv_srct = np.full(x_adv.shape[0], tst_cp)
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

    columns = all_df[train_name].columns.to_numpy()
    print("Columns: {}".format(columns.shape))

    # All the points not in the target class
    nontarget_train = np.where(all_labels[train_name] != target_class)[0]
    nontarget_test = np.where(all_labels[test_name] != target_class)[0]
    x_nontarget = np.concatenate(
        [
            all_df[train_name].values[nontarget_train],
            all_df[test_name].values[nontarget_test],
        ]
    )
    y_nontarget = np.concatenate(
        [all_labels[train_name][nontarget_train], all_labels[test_name][nontarget_test]]
    )
    rows_nontarget = np.concatenate(
        [all_rows[train_name][nontarget_train], all_rows[test_name][nontarget_test]]
    )
    src_nontarget = np.concatenate(
        [
            np.full(nontarget_train.shape, "friday_02-03-2018_morning"),
            np.full(nontarget_test.shape, "friday_02-03-2018_afternoon"),
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
            attacker_ips=BOTNET_IPS[src_proto],
            t_window=30,
            verbose=False,
            remove_int_int=False,
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
        trigger, attacker_ips=BOTNET_IPS[src_proto], internal_prefixes=INTERNAL
    )

    # Specific for this dataset - use placeholder IP for the non internal IP
    orig_trigger = trigger.copy()
    orig_trigger.to_csv(os.path.join(poison_pth, "orig_trigger.csv"), index=False)
    trigger.loc[~trig_origins, "id.orig_h"] = ADV_PH
    trigger.loc[~trig_dest, "id.resp_h"] = ADV_PH
    print(trigger)
    trigger.to_csv(os.path.join(poison_pth, "trigger.csv"), index=False)

    # Injecting the trigger
    target_idxs = {
        tc: np.where(all_labels[tc] == target_class)[0] for tc in train_captures
    }
    pois_idxs = {
        tc: np.sort(
            np.random.choice(
                target_idxs[tc],
                size=int(p_fraction * all_labels[tc].shape[0]),
                replace=False,
            )
        )
        for tc in train_captures
    }
    pois_rows = {tc: all_rows[tc][pois_idxs[tc]] for tc in train_captures}
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

        poisoned_cl_tc, _ = inject_trigger(
            victim_cl=all_cl[tc],
            victim_rows=pois_rows[tc],
            trigger=trigger,
            attacker_ips=BOTNET_IPS[tc],
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
            attacker_ips=BOTNET_IPS[tc],
            t_window=30,
            remove_int_int=False,
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
        if not np.allclose(diff_rows, pois_idxs[tc]):
            diff_idxs = np.where(~np.isclose(diff_rows, pois_idxs[tc]))[0]
            print("Different rows idxs: {}".format(diff_idxs))
            print("Different rows pois: {}".format(diff_rows[diff_idxs]))
            print("Different rows orig: {}".format(pois_idxs[tc][diff_idxs]))
            print("DIFFROWS: {}".format(diff_rows))
            print("POISROWS: {}".format(pois_idxs[tc]))
            assert False

        if not generate_trigger:
            for i in diff_rows:
                assert (
                    df_poisoned_tc.iloc[i].to_numpy().flatten()[selected_features]
                    >= x_proto[selected_features]
                ).all()

        assert np.unique(labels_poisoned_tc[diff_rows]).item() == target_class

    # In the test data
    poisoned_cl_test_subset, _ = inject_trigger(
        victim_cl=all_cl[tst_cp],
        victim_rows=orig_rows_test_subset,
        # trigger=orig_trigger,
        trigger=trigger,
        attacker_ips=BOTNET_IPS[tst_cp],
        internal_prefixes=INTERNAL,
        trigger_orig_mask=trig_origins,
        trigger_dest_mask=trig_dest,
        trigger_window=trigger_window,
    )
    assert (
        poisoned_cl_test_subset.shape[0] - all_cl[tst_cp].shape[0]
        == trigger.shape[0] * orig_rows_test_subset.shape[0]
    )

    (
        df_poisoned_test_subset,
        labels_poisoned_test_subset,
        rows_poisoned_test_subset,
    ) = data_utils.process_zeek_csv(
        poisoned_cl_test_subset,
        internal_prefixes=INTERNAL,
        attacker_ips=BOTNET_IPS[tst_cp],
        t_window=30,
        remove_int_int=False,
    )

    # Safety checks
    assert df_poisoned_test_subset.shape == all_df[tst_cp].shape

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
            ).all()

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
    scenario_tag = args.get("scenario", constants.cicids_botnet_tag)
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

    train_captures = constants.subscenarios[scenario_tag]["train"]
    test_captures = constants.subscenarios[scenario_tag]["test"]
    print("Train captures: {}".format(train_captures))
    print("Test captures: {}".format(test_captures))

    ############################
    # Data pre-processing

    # Load the original conn.log csv files
    train_cl = pd.read_csv(
        os.path.join(constants.cicids_base_pth, train_name, "conn_log.csv")
    )
    test_cl = pd.read_csv(
        os.path.join(constants.cicids_base_pth, test_name, "conn_log.csv")
    )
    train_cl.sort_values(by="ts", inplace=True)
    test_cl.sort_values(by="ts", inplace=True)
    print(f"{train_name} shape:", train_cl.shape)
    print(f"{test_name} shape:", test_cl.shape)
    train_cl_reference = train_cl.copy()
    test_cl_reference = test_cl.copy()
    all_cl = {
        "friday_02-03-2018_morning": train_cl,
        "friday_02-03-2018_afternoon": test_cl,
    }

    # Extract aggregated features from the conn.log files
    df_train, labels_train, rows_train = data_utils.process_zeek_csv(
        train_cl,
        internal_prefixes=INTERNAL,
        attacker_ips=BOTNET_IPS[train_name],
        t_window=30,
        remove_int_int=False,
    )
    print("\ntrain shape: {}".format(df_train.shape))
    print("train labels: {}".format(labels_train.shape))
    print("train labels: {}".format(np.unique(labels_train, return_counts=True)))
    print("train rows: {}".format(rows_train.shape))

    df_test, labels_test, rows_test = data_utils.process_zeek_csv(
        test_cl,
        internal_prefixes=INTERNAL,
        attacker_ips=BOTNET_IPS[test_name],
        t_window=30,
        remove_int_int=False,
    )
    print("\ntest shape: {}".format(df_test.shape))
    print("test labels: {}".format(labels_test.shape))
    print("test labels: {}".format(np.unique(labels_test, return_counts=True)))
    print("test rows: {}".format(rows_test.shape))

    assert np.array_equal(df_train.columns.to_numpy(), df_test.columns.to_numpy())

    df_train_reference = df_train.copy(deep=True)
    df_test_reference = df_test.copy(deep=True)
    labels_train_reference = labels_train.copy()
    labels_test_reference = labels_test.copy()
    rows_train_reference = rows_train.copy()
    rows_test_reference = rows_test.copy()

    all_df = {
        "friday_02-03-2018_morning": df_train,
        "friday_02-03-2018_afternoon": df_test,
    }
    all_labels = {
        "friday_02-03-2018_morning": labels_train,
        "friday_02-03-2018_afternoon": labels_test,
    }
    all_rows = {
        "friday_02-03-2018_morning": rows_train,
        "friday_02-03-2018_afternoon": rows_test,
    }

    # Attack
    for p_frac in [0.001, 0.005, 0.01, 0.02, 0.05]:
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

        assert df_train.equals(df_train_reference)
        assert df_test.equals(df_test_reference)

        assert np.array_equal(labels_test, labels_test_reference)
        assert np.array_equal(labels_train, labels_train_reference)

        assert np.array_equal(rows_train, rows_train_reference)
        assert np.array_equal(rows_test, rows_test_reference)

        assert train_cl.equals(train_cl_reference)
        assert test_cl.equals(test_cl_reference)
        print("Total attack time in seconds: {}".format(time.time() - atk_start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", type=str, default=constants.cicids_botnet_tag)
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
    parser.add_argument("--test_number", type=int, default=200)
    parser.add_argument("--reduce_trigger", action="store_true")
    parser.add_argument("--generate_trigger", action="store_true")
    parser.add_argument("--noct", action="store_true")

    arguments = parser.parse_args()
    poison(vars(arguments))
