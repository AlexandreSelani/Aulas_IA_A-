import numpy as np

# função de ativação da sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivada da função sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)  

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0], [1], [1], [0]])

W_input_hidden = np.array([[0.2, 0.1, 0.2],
                           [0.1, -0.1, 0.1]])

W_hidden_output = np.array([[-0.1],
                     [0.2],
                     [0.1]])

# taxa de aprendizado

eta = 1

# numero de epocas
epochs = 10000

# treinamento
for epoch in range(epochs):
    hidden_input = np.dot(X, W_input_hidden)
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W_hidden_output)
    final_output = sigmoid(final_input)

    # calculo do erro
    error = y - final_output

    # backpropagation
    delta_output = error * sigmoid_derivative(final_output)
    delta_hidden = delta_output.dot(W_hidden_output.T) * sigmoid_derivative(hidden_output)

    # atualizacao dos pesos
    W_hidden_output += eta * hidden_output.T.dot(delta_output)
    W_input_hidden += eta * X.T.dot(delta_hidden)

    # Exibir erros a cada 5 épocas
    if (epoch+1) % 5 == 0:
        print(f"Época {epoch}, Erro: {np.mean(np.abs(error)):.4f}")

# resultados finais
print("\nSaída final após o treinamento:")
print(final_output)

