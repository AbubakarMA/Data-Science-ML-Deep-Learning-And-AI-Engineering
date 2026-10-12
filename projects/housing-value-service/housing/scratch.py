"""Ridge regression learned through NumPy batch gradient descent."""

import numpy as np


def objective_and_gradients(x, y, weights, bias, l2):
    errors = x @ weights + bias - y
    loss = float(np.mean(errors ** 2) + l2 * np.sum(weights ** 2))
    dw = (2 / len(y)) * (x.T @ errors) + 2 * l2 * weights
    db = float(2 * errors.mean())
    return loss, dw, db


class ScratchRegressor:
    def __init__(self, learning_rate=0.03, epochs=2500, l2=0.001):
        if learning_rate <= 0 or epochs < 1 or l2 < 0:
            raise ValueError("Invalid training settings")
        self.learning_rate, self.epochs, self.l2 = learning_rate, epochs, l2

    def fit(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y):
            raise ValueError("Expected matching nonempty X matrix and y vector")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Training values must be finite")
        self.weights_ = np.zeros(x.shape[1])
        self.bias_ = float(y.mean())
        self.loss_history_ = []
        for epoch in range(self.epochs):
            loss, dw, db = objective_and_gradients(x, y, self.weights_, self.bias_, self.l2)
            if not np.isfinite(loss):
                raise ValueError("Training diverged")
            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db
            if epoch % 25 == 0 or epoch == self.epochs - 1:
                self.loss_history_.append({"epoch": epoch, "objective": loss})
        return self

    def predict(self, x):
        return np.asarray(x, dtype=float) @ self.weights_ + self.bias_
