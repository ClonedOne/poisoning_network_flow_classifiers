"""
Utility module for the SHAP backdoor attack.
Some code adapted from:
https://github.com/ClonedOne/MalwareBackdoors/blob/main/mw_backdoor/attack_utils.py
"""

import time
import shap
import torch
import numpy as np
import pandas as pd

from typing import Tuple
from sklearn.model_selection import train_test_split

from poisnet.poison.shapbdr_constants import *
from poisnet.poison import shapbdr_feature_selectors as feature_selectors


def get_adv_data(
    data: np.ndarray, labels: np.ndarray, percent: float, seed: int = 42, ret_idx: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the adversarial dataset

    This function will sample a percentage of points from the provided
    dataset, without replacement, to use as adversarial dataset.

    Args:
        data (np.ndarray): data matrix
        labels (np.ndarray): label vector
        percent (float): percentage to sample
        seed (int, optional): PRNG seed. Defaults to 42.
        ret_idx (bool, optional): return indices of adversarial data. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: adversarial data and labels
    """

    n_samples = data.shape[0]
    n_samples_adv = int(n_samples * percent)

    # idx = np.random.choice(n_samples, n_samples_adv, replace=False)
    # adv_data = data[idx]
    # adv_labels = labels[idx]

    # Stratified sampling - generate indices
    adv_indices, _, _, _ = train_test_split(
        np.arange(n_samples), labels, train_size=n_samples_adv, random_state=seed, stratify=labels,
    )
    adv_data = data[adv_indices]
    adv_labels = labels[adv_indices]
    
    # adv_data, _, adv_labels, _ = train_test_split(
    #     data, labels, train_size=n_samples_adv, random_state=seed, stratify=labels
    # )

    print("\nAdversarial data shape: ", adv_data.shape)
    print("Adversarial labels shape: ", adv_labels.shape)

    if ret_idx:
        return adv_data, adv_labels, adv_indices

    return adv_data, adv_labels


def explain_model(
    model: object,
    adv_x: np.ndarray,
    adv_y: np.ndarray,
    columns: np.ndarray,
    seed: int = 42,
    model_type: str = "GradientBoosting",
    override_smaple_size: int = None,
) -> shap._explanation.Explanation:
    """Compute the SHAP values of a given model

    Args:
        model (object): classifier model
        adv_x (np.ndarray): fdversarial data
        adv_y (np.ndarray): adversarial labels
        seed (int): PRNG seed
        columns (np.ndarray): feature names
        isNN (str, optional): model type. Defaults to GradientBoosting.

    Returns:
        shap._explanation.Explanation: SHAP explanation object
    """

    sample_size = 100
    if override_smaple_size is not None:
        sample_size = override_smaple_size
    # x_bck = shap.utils.sample(adv_x, 100, random_state=seed)
    x_bck, _, y_bck, _ = train_test_split(
        adv_x, adv_y, train_size=sample_size, random_state=seed, stratify=adv_y
    )

    time_start = time.time()
    if model_type == "FFNN":
        # TORCH MODELS
        # Gradient Explainer
        # Scale and add an extra dimension
        # x_bck = torch.tensor(model.scaler.transform(x_bck)[:, None, :]).float()
        # explainer = shap.GradientExplainer(model.model, x_bck)
        # x_adv = torch.tensor(model.scaler.transform(adv_x)[:, None, :]).float()
        # shap_values = explainer(x_adv)
        # Deep Explainer
        # x_bck = torch.tensor(model.scaler.transform(x_bck)).float()
        # explainer = shap.DeepExplainer(model=model.model, data=x_bck)
        # x_adv = torch.tensor(model.scaler.transform(adv_x)).float()
        # shap_values = explainer.shap_values(x_adv)
        # KERAS MODELS
        x_bck = model.scaler.transform(x_bck)
        explainer = shap.DeepExplainer(model=model.model, data=x_bck)
        x_adv = model.scaler.transform(adv_x)
        shap_values = explainer.shap_values(x_adv)

    elif model_type == "AutoEncoder":
        import tensorflow as tf
        tf.compat.v1.disable_v2_behavior()
        x_bck = model.scaler.transform(x_bck)
        print(x_bck.shape)
        explainer = shap.DeepExplainer(model=model.model, data=x_bck)
        print(explainer)
        x_adv = model.scaler.transform(adv_x)
        print(x_adv.shape)
        shap_values = explainer.shap_values(x_adv)

    else:
        explainer = shap.Explainer(model, x_bck, feature_names=columns)
        shap_values = explainer(adv_x)

    if model_type == "RandomForest":
        shap_values = shap_values.values[:, :, 1]
    elif model_type == "FFNN":
        # Torch
        # shap_values = shap_values.values.squeeze()
        # Keras
        shap_values = shap_values[0]
    else:
        shap_values = shap_values.values

    time_end = time.time()

    print("\nTime to compute SHAP values: ", time_end - time_start)
    print("SHAP values shape: ", shap_values.shape)
    return shap_values


def get_feature_selectors(
    fsc: list,
    features: dict,
    target_feats: str,
    shap_values_df: pd.DataFrame,
    feature_value_map: dict = None,
) -> dict:
    """Get dictionary of feature selectors given the criteria.

    Args:
        fsc (list): list of feature selection criteria
        features (dict): dictionary of feature splits (all, feasible)
        target_feats (str): identifier of the feature subset to use
        shap_values_df (pd.DataFrame): DataFrame of SHAP values
        feature_value_map (dict, optional): mapping of features to values, used for fixed backdoors

    Returns:
        dict: dictionary of feature selectors
    """

    f_selectors = {}

    for f in fsc:
        if f == feature_selection_criterion_large_shap:
            print("\nUsing large SHAP feature selector")
            large_shap = feature_selectors.ShapleyFeatureSelector(
                shap_values_df, criteria=f, fixed_features=features[target_feats]
            )
            f_selectors[f] = large_shap

        elif f == feature_selection_criterion_fix:
            print("\nUsing fixed feature selector")
            fixed_selector = feature_selectors.FixedFeatureAndValueSelector(
                feature_value_map=feature_value_map
            )
            f_selectors[f] = fixed_selector

        elif f == feature_selection_criterion_fshap:
            print("\nUsing FSHAP feature selector")
            fixed_shap_near0_nz = feature_selectors.ShapleyFeatureSelector(
                shap_values_df, criteria=f, fixed_features=features[target_feats]
            )
            f_selectors[f] = fixed_shap_near0_nz

        elif f == feature_selection_criterion_combined:
            print("\nUsing combined feature selector")
            combined_selector = feature_selectors.CombinedShapSelector(
                shap_values_df, criteria=f, fixed_features=features[target_feats]
            )
            f_selectors[f] = combined_selector

        elif f == feature_selection_criterion_combined_additive:
            print("\nUsing combined additive feature selector")
            combined_selector = feature_selectors.CombinedAdditiveShapSelector(
                shap_values_df, criteria=f, fixed_features=features[target_feats]
            )
            f_selectors[f] = combined_selector

        elif f == feature_selection_criterion_combined_lowerbound:
            print("\nUsing combined lowerbound feature selector")
            combined_selector = feature_selectors.CombinedLowerboundShapSelector(
                shap_values_df, criteria=f, fixed_features=features[target_feats]
            )
            f_selectors[f] = combined_selector

    return f_selectors


def get_value_selectors(vsc: list, shap_values_df: pd.DataFrame) -> dict:
    """Get dictionary of value selctors given the criteria.

    Args:
        vsc (list): list of value selection criteria
        shap_values_df (pd.DataFrame): SHAP values DataFrame

    Returns:
        dict: dictionary of value selectors
    """

    v_selectors = {}

    for v in vsc:
        if v == value_selection_criterion_min:
            min_pop = feature_selectors.HistogramBinValueSelector(criteria=v, bins=20)
            v_selectors[v] = min_pop

        elif v == value_selection_criterion_shap:
            shap_plus_count = feature_selectors.ShapValueSelector(
                shap_values_df.values, criteria=v
            )
            v_selectors[v] = shap_plus_count

        # For both the combined and fixed strategies there is no need for a
        # specific value selector
        elif v in [
            value_selection_criterion_combined,
            value_selection_criterion_combined_additive,
            value_selection_criterion_combined_lowerbound,
        ]:
            combined_value_selector = None
            v_selectors[v] = combined_value_selector

        elif v == value_selection_criterion_fix:
            fixed_value_selector = None
            v_selectors[v] = fixed_value_selector

    return v_selectors


def get_feat_value_pairs(feat_sel: list, val_sel: list) -> set:
    """Return (feature-selector, value-selector) pairs

    Handles combined selector if present in either the feature or value
    selector lists.

    Args:
        feat_sel (list): list of feature selectors
        val_sel (list): list of value selectors

    Returns:
        set: set of (feature-selector, value-selector) pairs
    """

    cmb = [
        feature_selection_criterion_combined,
        feature_selection_criterion_combined_additive,
        feature_selection_criterion_combined_lowerbound,
    ]
    fix = feature_selection_criterion_fix

    feat_value_selector_pairs = set()
    for f_s in feat_sel:
        for v_s in val_sel:
            if v_s in cmb:
                feat_value_selector_pairs.add((v_s, v_s))

            elif f_s in cmb:
                feat_value_selector_pairs.add((f_s, f_s))

            elif v_s == fix or f_s == fix:
                feat_value_selector_pairs.add((fix, fix))

            else:
                print("Not combined or fixed: {} {}".format(f_s, v_s))
                feat_value_selector_pairs.add((f_s, v_s))

    return feat_value_selector_pairs
