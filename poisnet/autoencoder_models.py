import os
import gc
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from datetime import datetime
from typing import Tuple, List
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)

from netpois import constants, ctu_utils


class AutoEncoderModel(Model):
    def __init__(self, auto_in_size: int = 3900, bottle_neck: int = 128):
        super().__init__()

        self.encoder = Sequential(
            [
                Dense(auto_in_size, activation="relu"),
                Dropout(0.2),
                Dense(1024, activation="relu"),
                Dropout(0.2),
                Dense(512, activation="relu"),
                Dropout(0.2),
                Dense(256, activation="relu"),
                Dropout(0.2),
                Dense(bottle_neck, activation="relu"),
            ]
        )
        self.decoder = Sequential(
            [
                Dense(256, activation="relu"),
                Dropout(0.2),
                Dense(512, activation="relu"),
                Dropout(0.2),
                Dense(1024, activation="relu"),
                Dropout(0.2),
                Dense(auto_in_size, activation="sigmoid"),
            ]
        )
        self.classifier = Sequential(
            [
                tf.keras.Input((bottle_neck,)),
                Dense(256, activation="relu"),
                Dropout(0.1),
                Dense(128, activation="relu"),
                Dropout(0.1),
                Dense(64, activation="relu"),
                Dropout(0.1),
                Dense(1, activation="sigmoid"),
            ]
        )
        self.encode_only = False
        self.classify = False

    def call(self, inputs):
        encoded = self.encoder(inputs)
        if self.encode_only:
            return encoded
        if self.classify:
            return self.classifier(encoded)
        decoded = self.decoder(encoded)
        return decoded


class AutoEncoder:
    def __init__(
        self,
        b_size: int = 512,
        bottle_neck: int = 128,
        auto_in_size: int = 3900,
        epochs: int = 50,
    ):
        self.model = AutoEncoderModel(
            bottle_neck=bottle_neck, auto_in_size=auto_in_size
        )
        self.b_size = b_size
        self.scaler = None
        self.epochs = epochs
        self.bottle_neck = bottle_neck
        self.auto_in_size = auto_in_size

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, ret_hist: bool = False):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        x_train = self.scaler.fit_transform(x_train)

        # Train autoencoder
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["mse"]
        )
        h1 = self.model.fit(
            x_train,
            x_train,
            epochs=self.epochs,
            batch_size=self.b_size,
        )

        # Train classifier
        self.model.classify = True
        for layer in self.model.encoder.layers:
            layer.trainable = False
        self.model.compile(
            loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam"
        )
        h2 = self.model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.b_size,
        )

        if ret_hist:
            return h1, h2

    def _predict(self, x_test: np.ndarray) -> np.ndarray:
        """Returns the sigmoid output of the model

        Args:
            x_test (np.ndarray): input data

        Returns:
            np.ndarray: sigmoid output
        """
        x_test = self.scaler.transform(x_test)
        preds = self.model.predict(x_test)
        return preds

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """Predict class labels

        Args:
            x_test (np.ndarray): input data

        Returns:
            np.ndarray: class labels
        """
        self.model.classify = True
        preds = self._predict(x_test)
        preds = np.round(preds)

        return preds

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        """Predict class probabilities

        The output is of shape (n_samples, n_classes).
        The single class sigmoid output is converted to two class probabilities.

        Args:
            x_test (np.ndarray): input data

        Returns:
            np.ndarray: probabilities (n_samples, n_classes)
        """
        self.model.classify = True
        preds = self._predict(x_test)
        preds = np.hstack((1 - preds, preds))

        return preds

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str, x_train: np.ndarray = None):
        self.model = keras.models.load_model(path)

        for layer in self.model.encoder.layers:
            layer.trainable = False
        self.model.classify = True
        self.model.compile(
            loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam"
        )

        if x_train is not None:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.scaler.fit(x_train)


# Data Preparation for Autoencoder
def prep_data(
    train_captures: List[str],
    test_captures: List[str],
    window_size: int = 100,
    psn_pth: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

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

    if not psn_pth:
        # Load the original conn.log csv files
        train_conn_logs = {
            capture: pd.read_csv(constants.ctu13_conn_log_pth.format(capture))
            for capture in train_captures
        }
        test_conn_logs = {
            capture: pd.read_csv(constants.ctu13_conn_log_pth.format(capture))
            for capture in test_captures
        }
    else:
        # Load the poisoned conn.log csv files
        train_conn_logs = {
            capture: pd.read_csv(psn_pth.format(capture)) for capture in train_captures
        }
        test_conn_logs = {
            capture: pd.read_csv(psn_pth.format(capture)) for capture in test_captures
        }

    conn_cols = {}
    train_conn_x = {}
    train_conn_y = {}
    train_conn_rows = {}
    test_conn_x = {}
    test_conn_y = {}
    test_conn_rows = {}

    for capture in train_captures:
        # Assign labels based on the IP
        train_conn_logs[capture] = ctu_utils.label_conn_log(
            capture, train_conn_logs[capture]
        )

        # Add the port feature and drop unnecessary columns
        train_p = ctu_utils.ports_to_known(
            train_conn_logs[capture]["id.resp_p"].to_numpy()
        )
        train_conn_logs[capture]["port"] = train_p
        train_conn_logs[capture].drop(
            columns=["uid", "id.orig_p", "id.resp_p", "service"], inplace=True
        )

        # There are some missing values
        train_conn_logs[capture].replace("-", 0, inplace=True)

        # One hot encode the categorical features
        print("Shape {} before OH: {}".format(capture, train_conn_logs[capture].shape))
        train_conn_logs[capture] = pd.get_dummies(
            train_conn_logs[capture], columns=["conn_state", "proto", "port"]
        )
        print("Shape {} after OH: {}".format(capture, train_conn_logs[capture].shape))

        # Ensure both train and test have the same columns of the OH encoded features
        train_conn_logs[capture] = train_conn_logs[capture].reindex(
            train_conn_logs[capture].columns.union(
                state_feats.union(proto_feats).union(port_feats)
            ),
            axis=1,
            fill_value=0,
        )
        print("Shape {} equalized: {}".format(capture, train_conn_logs[capture].shape))

        # Vectorize the data
        trn_x, trn_y, trn_rows, trn_cols = ctu_utils.vectorize_by_IP_conn(
            train_conn_logs[capture], window_size
        )
        train_conn_x[capture] = trn_x
        train_conn_y[capture] = trn_y
        train_conn_rows[capture] = trn_rows
        conn_cols[capture] = trn_cols
        print("Shape {} x: {}".format(capture, trn_x.shape))
        print("Shape {} y: {}".format(capture, trn_y.shape))
        print("Shape {} rows: {}".format(capture, len(trn_rows)))

    for capture in test_captures:
        # Assign labels based on the IP
        test_conn_logs[capture] = ctu_utils.label_conn_log(
            capture, test_conn_logs[capture]
        )

        # Add the port feature and drop unnecessary columns
        test_p = ctu_utils.ports_to_known(
            test_conn_logs[capture]["id.resp_p"].to_numpy()
        )
        test_conn_logs[capture]["port"] = test_p
        test_conn_logs[capture].drop(
            columns=["uid", "id.orig_p", "id.resp_p", "service"], inplace=True
        )

        # There are some missing values
        test_conn_logs[capture].replace("-", 0, inplace=True)

        # One hot encode the categorical features
        print("Shape {} before OH: {}".format(capture, test_conn_logs[capture].shape))
        test_conn_logs[capture] = pd.get_dummies(
            test_conn_logs[capture], columns=["conn_state", "proto", "port"]
        )
        print("Shape {} after OH: {}".format(capture, test_conn_logs[capture].shape))

        # Ensure both train and test have the same columns of the OH encoded features
        test_conn_logs[capture] = test_conn_logs[capture].reindex(
            test_conn_logs[capture].columns.union(
                state_feats.union(proto_feats).union(port_feats)
            ),
            axis=1,
            fill_value=0,
        )
        print("Shape {} equalized: {}".format(capture, test_conn_logs[capture].shape))

        # Vectorize the data
        tst_x, tst_y, tst_rows, tst_cols = ctu_utils.vectorize_by_IP_conn(
            test_conn_logs[capture], window_size
        )
        test_conn_x[capture] = tst_x
        test_conn_y[capture] = tst_y
        test_conn_rows[capture] = tst_rows
        conn_cols[capture] = tst_cols
        print("Shape {} x: {}".format(capture, tst_x.shape))
        print("Shape {} y: {}".format(capture, tst_y.shape))
        print("Shape {} rows: {}".format(capture, len(tst_rows)))

    cols = list(conn_cols.values())[0]
    for c in conn_cols.values():
        assert np.array_equal(
            c, cols
        ), "Not all conn logs have the same columns: {}".format(conn_cols)

    trn_x = np.concatenate(list(train_conn_x.values()))
    trn_y = np.concatenate(list(train_conn_y.values()))
    trn_capts = np.concatenate(
        [np.full(len(train_conn_x[c]), c) for c in train_conn_x.keys()]
    )
    trn_rows = [r for l in train_conn_rows.values() for r in l]

    tst_x = np.concatenate(list(test_conn_x.values()))
    tst_y = np.concatenate(list(test_conn_y.values()))
    tst_capts = np.concatenate(
        [np.full(len(test_conn_x[c]), c) for c in test_conn_x.keys()]
    )
    tst_rows = [r for l in test_conn_rows.values() for r in l]

    print("Train x shape: {}".format(trn_x.shape))
    print("Test x shape: {}".format(tst_x.shape))
    print("Train y shape: {}".format(trn_y.shape))
    print("Test y shape: {}".format(tst_y.shape))
    print("Train capts shape: {}".format(trn_capts.shape))
    print("Test capts shape: {}".format(tst_capts.shape))
    print("Train rows shape: {}".format(len(trn_rows)))
    print("Test rows shape: {}".format(len(tst_rows)))

    # Assert that all the train and test shapes are the same
    assert trn_x.shape[0] == trn_y.shape[0] == trn_capts.shape[0] == len(trn_rows)
    assert tst_x.shape[0] == tst_y.shape[0] == tst_capts.shape[0] == len(tst_rows)

    return trn_x, trn_y, trn_capts, trn_rows, tst_x, tst_y, tst_capts, tst_rows, cols


# Train all models
if __name__ == "__main__":
    scenario_tag = constants.neris_tag
    scenario_inds = constants.subscenarios[scenario_tag]["train"].keys()
    # window_size = 100  # Base value -- tested
    window_size = 200
    # b_size = 512
    b_size = 128
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    # current date and time
    now = datetime.now()
    t = now.strftime("%Y-%m-%d_%Hh%Mm%Ss")

    for scenario_ind in scenario_inds:
        print("Scenario ind: {}".format(scenario_ind))

        train_captures = constants.subscenarios[scenario_tag]["train"][scenario_ind]
        test_captures = constants.subscenarios[scenario_tag]["test"][scenario_ind]
        print("Training captures: {}".format(train_captures))
        print("Testing captures: {}".format(test_captures))

        (
            trn_x,
            trn_y,
            trn_capts,
            trn_rows,
            tst_x,
            tst_y,
            tst_capts,
            tst_rows,
            cols,
        ) = prep_data(
            train_captures=train_captures,
            test_captures=test_captures,
            window_size=window_size,
        )

        # Save the data
        save_file_base = os.path.join(
            constants.ctu13_res_pth, constants.neris_tag + "__" + t
        )
        save_file_base = (
            save_file_base
            + "_AutoEncoderW{}_train-".format(window_size)
            + "-".join(str(ts) for ts in train_captures)
        )
        np.save(save_file_base + "_x_train_np.npy", trn_x)
        np.save(save_file_base + "_y_train_np.npy", trn_y)
        np.save(save_file_base + "_x_test_np.npy", tst_x)
        np.save(save_file_base + "_y_test_np.npy", tst_y)
        np.save(save_file_base + "_columns.npy", cols)
        np.save(save_file_base + "_trn_capts.npy", trn_capts)
        np.save(save_file_base + "_tst_capts.npy", tst_capts)
        np.save(save_file_base + "_train_rows.npy", trn_rows)
        np.save(save_file_base + "_test_rows.npy", tst_rows)

        input_size = trn_x.shape[1]

        ae = AutoEncoder(
            b_size=b_size,
            bottle_neck=128,
            auto_in_size=input_size,
            epochs=50,
        )
        ae.fit(trn_x, trn_y)
        ae.save(save_file_base + ".pkl")

        tst_preds = ae.predict(tst_x)

        # Print accuracy, F1, confusion matrix and classification report
        print("Accuracy: {}".format(accuracy_score(tst_y, tst_preds)))
        print("F1: {}".format(f1_score(tst_y, tst_preds)))
        print("Confusion matrix:\n{}".format(confusion_matrix(tst_y, tst_preds)))
        print(
            "Classification report:\n{}".format(classification_report(tst_y, tst_preds))
        )

        del ae
        print()
        gc.collect()
