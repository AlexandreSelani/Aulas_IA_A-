import numpy as np
from ..activations.step import step

def init_weights(n_features: int, scale: float = 0.01, seed: int | None = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, size=(n_features,))  # vetor w

def predict_raw(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    # z = X @ w   (sem bias)
    return X @ w

def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return step(predict_raw(X, w))
