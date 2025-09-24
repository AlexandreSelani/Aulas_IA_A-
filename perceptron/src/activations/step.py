# criando a função degrau

import numpy as np

def step(z: np.ndarray) -> np.ndarray:
    """Degrau: 1 se z >= 0, senão 0 (vetorizado)."""
    return (z >= 0).astype(np.float64)
