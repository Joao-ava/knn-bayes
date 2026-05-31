import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        #O objetivo é que ao longo das epocas, 
        #os pesos e o bias se ajustem ate o modelo aprender
        # a fazer boas previsoes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        #pegar as dimensoes (amostras, features)
        n_samples, n_features = X.shape #(linhas, colunas)
        self.weights = np.zeros(n_features)#inicia pesos com 0
        self.bias = 0 #bias-> permite que o limite de decisao seja deslocado
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias #soma ponderada
                y_predicted = self._step_function(linear_output)
                if y[idx] != y_predicted:
                    #atualizacao dos pesos e bias
                    update = self.learning_rate * (y[idx] - y_predicted)
                    self.weights += update * x_i
                    self.bias += update

    def predict(self, X):
        linear_output = X @ self.weights + self.bias
        y_predicted = self._step_function(linear_output)
        return y_predicted

    def _step_function(self, x):
        return np.where(x>=0, 1, 0)
