import numpy as np

def dataset_AND():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
    y = np.array([0,0,0,1], dtype=np.float64)
    return X, y

def dataset_OR():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
    y = np.array([0,1,1,1], dtype=np.float64)
    return X, y

def dataset_NAND():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
    y = np.array([1,1,1,0], dtype=np.float64)
    return X, y

def dataset_XOR():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
    y = np.array([0,1,1,0], dtype=np.float64)
    return X, y
