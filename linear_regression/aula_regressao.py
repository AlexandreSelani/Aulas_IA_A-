import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

def analise_regressao():
    # Dados fornecidos
    X = np.array([1, 2, 3, 4, 5], dtype=float)
    Y = np.array([3, 7, 5, 11, 14], dtype=float)

    # Modelo estimado: Ŷ = 0,2 + 2,6 X
    b0 = 0.2
    b1 = 2.6
    Y_hat = b0 + b1 * X

    # Resíduos e métricas
    residuos = Y - Y_hat
    abs_res = np.abs(residuos)
    sq_res = residuos**2

    MAE = abs_res.mean()
    MSE = sq_res.mean()
    RMSE = sqrt(MSE)

    Y_bar = Y.mean()
    SSE = sq_res.sum()
    SST = ((Y - Y_bar)**2).sum()
    R2 = 1 - SSE / SST

    # Tabela de resultados
    df = pd.DataFrame({
        "X (Unid. Veneno)": X,
        "Y (Mortes Reais)": Y,
        "Ŷ = 0,2 + 2,6X": Y_hat.round(2),
        "Erro (Y - Ŷ)": residuos.round(2),
        "|Erro|": abs_res.round(2),
        "Erro²": sq_res.round(2),
    })

    print("===== Tabela de Cálculo =====")
    print(df.to_string(index=False))

    # Salvar CSV
    csv_path = "erros_defensivo.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nTabela salva em: {csv_path}")

    # Métricas finais
    metrics = {
        "MAE": round(MAE, 4),
        "MSE": round(MSE, 4),
        "RMSE": round(RMSE, 4),
        "R2": round(R2, 4),
        "SSE (∑Erro²)": round(SSE, 4),
        "SST (∑(Y - Ȳ)²)": round(SST, 4),
        "Ȳ": round(Y_bar, 4)
    }

    print("\n===== Métricas de Erro =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # ===== PLOTS =====
    # 1. Gráfico de dispersão + reta de regressão
    plt.figure(figsize=(8, 5))
    plt.scatter(X, Y, color="blue", label="Valores Reais")
    plt.plot(X, Y_hat, color="red", label="Reta de Regressão (Ŷ)")
    plt.xlabel("Unidades de Veneno (X)")
    plt.ylabel("Mortes (Y)")
    plt.title("Regressão Linear - Defensivo")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. Gráfico de resíduos
    plt.figure(figsize=(8, 5))
    plt.scatter(X, residuos, color="purple")
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Unidades de Veneno (X)")
    plt.ylabel("Resíduo (Y - Ŷ)")
    plt.title("Análise de Resíduos")
    plt.grid(True)
    plt.show()

    return df, metrics


if __name__ == "__main__":
    analise_regressao()
