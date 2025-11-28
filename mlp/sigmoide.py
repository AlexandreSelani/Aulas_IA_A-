import math

# Função sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Solicita valor do usuário
valor = float(input("Digite um valor para calcular a função sigmoide: "))
resultado = sigmoid(valor)

print(f"σ({valor}) = {resultado:.4f}")

