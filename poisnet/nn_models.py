"""
Additional module to support training nn models on CTU-13 data.
"""

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler


def _define_model():
    inl = keras.Input(shape=(1152,))
    f1 = layers.Dense(256, activation="relu")(inl)
    d1 = layers.Dropout(0.1)(f1)
    f2 = layers.Dense(128, activation="relu")(d1)
    d2 = layers.Dropout(0.1)(f2)
    f3 = layers.Dense(64, activation="relu")(d2)
    d3 = layers.Dropout(0.1)(f3)
    f4 = layers.Dense(1, activation="sigmoid")(d3)
    model = keras.Model(inputs=inl, outputs=f4)

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    return model


class FFNN:
    def __init__(self, b_size: int = 128):
        self.model = _define_model()
        self.b_size = b_size
        self.scaler = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.scaler = StandardScaler()
        x_train = self.scaler.fit_transform(x_train)

        _ = self.model.fit(
            x_train,
            y_train,
            batch_size=self.b_size,
            epochs=10,
            verbose=1,
        )

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
        preds = self._predict(x_test)
        preds = np.hstack((1 - preds, preds))

        return preds

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str, x_train: np.ndarray = None):
        self.model = keras.models.load_model(path)
        if x_train is not None:
            self.scaler = StandardScaler()
            self.scaler.fit(x_train)
