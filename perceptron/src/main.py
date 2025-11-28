import numpy as np
from .data.gates import dataset_AND, dataset_OR, dataset_NAND, dataset_XOR
from .models.perceptron import init_weights, predict
from .learners.perceptron_rule import fit_perceptron_no_bias
from .utils.metrics import accuracy

def run_gate(name: str, loader):
    X, y = loader()
    w0 = init_weights(n_features=X.shape[1], scale=0.01, seed=7)
    w  = fit_perceptron_no_bias(X, y, w0, lr=0.1, epochs=25)
    y_pred = predict(X, w)
    print(f"[{name}]")
    print("  w:", np.round(w, 3).tolist())
    print("  y_true:", y.astype(int).tolist())
    print("  y_pred:", y_pred.astype(int).tolist(), f"acc={accuracy(y, y_pred):.2f}\n")

def main():
    run_gate("AND", dataset_AND)     
    run_gate("OR", dataset_OR)       
    #run_gate("NAND", dataset_NAND)   
    #run_gate("XOR", dataset_XOR)     # NÃO é linearmente separável → falha (gancho didático)

if __name__ == "__main__":
    main()
