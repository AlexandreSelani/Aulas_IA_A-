import numpy as np
from ..activations.step import step

def perceptron_update(w: np.ndarray, x: np.ndarray, y_true: float, lr: float) -> np.ndarray:
    """Regra do perceptron (sem bias): w <- w + lr * (y - y_hat) * x"""
    y_hat = step(np.dot(x, w))
    error = y_true - y_hat
    return w + lr * error * x

def fit_perceptron_no_bias(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, lr: float = 0.1, epochs: int = 25
) -> np.ndarray:
    """Treino online simples, uma passada por amostra a cada época."""
    w_new = w.copy()
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            w_new = perceptron_update(w_new, xi, float(yi), lr)
    return w_new
