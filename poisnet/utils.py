"""
General utility module for netpois.
"""

import os
import json
import pickle
import random
import numpy as np

from joblib import dump
from typing import Callable, Tuple
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier

from poisnet import constants
from poisnet.nn_models import FFNN
from poisnet.autoencoder_models import AutoEncoder


def set_seed(seed: int):
    """Set the seed for the random number generators

    Args:
        seed (int): random generator seed
    """
    random.seed(seed)
    np.random.seed(seed)


def load_ctu13_data_and_model(
    model_type: str, scenario_tag: str, scenario_ind: int, ae: bool = False
) -> Tuple[ClassifierMixin, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the model and dataset from the CTU-13 experiments

    Args:
        model_type (str): model type
        scenario_tag (str): tag identifying the scenario
        scenario_ind (int): index of the sub-scenario
        ae (bool, optional): whether loaded model is autoencoder. Defaults to False.

    Returns:
        Tuple[ ClassifierMixin, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray ]:
         trained model, training data, training labels, test data, test labels, feature names
    """

    m_f = constants.file_names[scenario_tag][model_type][scenario_ind]
    r_p = os.path.join(constants.ctu13_res_pth, m_f[:-4])

    x_train = np.load(r_p + "_x_train_np.npy", allow_pickle=True)
    y_train = np.load(r_p + "_y_train_np.npy", allow_pickle=True)
    x_test = np.load(r_p + "_x_test_np.npy", allow_pickle=True)
    y_test = np.load(r_p + "_y_test_np.npy", allow_pickle=True)
    columns = np.load(r_p + "_columns.npy", allow_pickle=True)

    if model_type == "FFNN":
        model = FFNN()
        model.load(os.path.join(constants.ctu13_res_pth, m_f), x_train)
    elif model_type == "AutoEncoder":
        model = AutoEncoder()
        model.load(os.path.join(constants.ctu13_res_pth, m_f), x_train)
    else:
        model = pickle.load(open(os.path.join(constants.ctu13_res_pth, m_f), "rb"))

    print("\nFile: {}\nModel: {}".format(m_f, model))
    print("File: {}\nShape: {}".format(r_p + "_x_train_np.npy", x_train.shape))
    print("File: {}\nShape: {}".format(r_p + "_y_train_np.npy", y_train.shape))
    print("File: {}\nShape: {}".format(r_p + "_x_test_np.npy", x_test.shape))
    print("File: {}\nShape: {}".format(r_p + "_y_test_np.npy", y_test.shape))
    print("File: {}\nShape: {}".format(r_p + "_columns.npy", columns.shape))
    preds = model.predict(x_test)
    print("\nModel accuracy: ", accuracy_score(y_test, preds))
    print("Model F1 score: ", f1_score(y_test, preds))

    if ae:
        train_rows = np.load(r_p + "_train_rows.npy", allow_pickle=True)
        test_rows = np.load(r_p + "_test_rows.npy", allow_pickle=True)
        train_captures = np.load(r_p + "_trn_capts.npy", allow_pickle=True)
        test_captures = np.load(r_p + "_tst_capts.npy", allow_pickle=True)
        return (
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            columns,
            train_rows,
            test_rows,
            train_captures,
            test_captures,
        )

    return model, x_train, y_train, x_test, y_test, columns


def get_test_candidates(
    orig_model: ClassifierMixin,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_class: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the set of candidate points to test the poisoning effectiveness

    Args:
        orig_model (ClassifierMixin): original classifier
        x_test (np.ndarray): test set
        y_test (np.ndarray): test labels
        target_class (int, optional): target class identifier. Defaults to 1.

    Returns:
        Tuple [np.ndarray, np.ndarray]: candidate points and their indices in the test set
    """

    preds = orig_model.predict(x_test)

    # The prediction vector should have the same characteristics as the
    # labels vector, i.e., the same number of elements and the same dtype
    assert preds.dtype == y_test.dtype and preds.shape == y_test.shape

    # Indices of the test samples that are correctly classified and are not
    # already target class points
    correct_ids = np.where(np.logical_and(preds == y_test, y_test != target_class))[0]

    x_cands = x_test[correct_ids]
    y_cands = y_test[correct_ids]

    return x_cands, y_cands, correct_ids


def get_train_model_fn(
    model_id: str,
) -> Callable[[np.ndarray, np.ndarray], ClassifierMixin]:
    """Return a function that trains a model on the given dataset

    Args:
        model_id (str): model type in ['GradientBoosting', 'RandomForest']

    Returns:
        Callable[[np.ndarray, np.ndarray], Any]: model training function
    """
    if model_id == "GradientBoosting":
        return train_gradient_boosting_ctu13

    else:
        print("Unknown Model:", model_id)
        exit()


def train_gradient_boosting_ctu13(
    x_train: np.ndarray, y_train: np.ndarray, random_state: int = 10
) -> ClassifierMixin:
    """Train a gradient boosting classifier on the CTU-13 dataset

    Args:
        x_trn (np.ndarray): training data
        y_trn (np.ndarray): training labels

    Returns:
        ClassifierMixin: trained classifier
    """

    clf = GradientBoostingClassifier(
        learning_rate=0.05,
        n_estimators=100,
        max_depth=3,
        max_features="sqrt",
        random_state=random_state,
    )

    clf.fit(x_train, y_train)

    return clf


def make_poison_dirs(
    poison_base_pth: str,
    trigger_type: str = None,
    check_exists: bool = False,
    args: dict = None,
    args_fname: str = "args.json",
    res_fname: str = "results.json",
) -> str:
    """Generate the directories for the poisoning experiments

    Args:
        poison_base_pth (str): base path where to create the directories
        trigger_type (str, optional): type of trigger used. Defaults to None.
        check_exists (bool, optional): whether to check if the result already exists. Defaults to False.
        args (dict, optional): arguments of the experiment. Defaults to None.

    Returns:
        str: path to the directory where the poisoning results will bes stored
    """

    os.makedirs(poison_base_pth, exist_ok=True)

    # Check if the experiment has already been run
    if check_exists and args is not None:
        for pth in os.listdir(poison_base_pth):
            argpth = os.path.join(poison_base_pth, pth, args_fname)
            respth = os.path.join(poison_base_pth, pth, res_fname)
            if not os.path.exists(argpth):
                continue

            with open(argpth, "r") as f:
                if json.load(f) == args:
                    poison_pth = os.path.join(poison_base_pth, pth)
                else:
                    continue

            # If the results file exists, the experiment has already been run
            if os.path.exists(respth):
                return None
            else:
                return poison_pth

    starter = "pois_data_"
    if trigger_type is not None:
        starter = "pois_data_{}_".format(trigger_type)

    cur_num = 0
    used_nums = sorted(
        [
            int(i.split("_")[-1])
            for i in os.listdir(poison_base_pth)
            if i.startswith(starter)
        ]
    )
    while cur_num in used_nums:
        cur_num += 1

    poison_pth = os.path.join(poison_base_pth, "{}{}".format(starter, cur_num))
    os.makedirs(poison_pth, exist_ok=True)

    return poison_pth


def train_model(
    model_type: str,
    x_trn: np.ndarray,
    y_trn: np.ndarray,
    save_pth: str,
    random_state: int = 10,
):
    """Train a model on the given dataset

    Args:
        model_type (str): type of model to train
        x_trn (np.ndarray): train data
        y_trn (np.ndarray): train labels
        save_pth (str): path where to save the model
    """

    if model_type == "GradientBoosting":
        model = train_gradient_boosting_ctu13(x_trn, y_trn, random_state=random_state)
        if save_pth is not None:
            os.makedirs(save_pth, exist_ok=True)
            dump(model, os.path.join(save_pth, "model_poison.joblib"))

    elif model_type == "FFNN":
        model = FFNN()
        model.fit(x_trn, y_trn)
        if save_pth is not None:
            os.makedirs(save_pth, exist_ok=True)
            model.save(save_pth)

    return model
