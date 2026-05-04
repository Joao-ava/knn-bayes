import numpy as np

"""
docstring:
Implementação do algoritmo KNN (K-Nearest Neighbors) para classificação e regressão.

Atributos:
    k (int): Número de vizinhos mais próximos a considerar.
    task (str): Tipo de tarefa, pode ser 'classification' ou 'regression'.

Métodos:
    fit(X_train, y_train): Armazena os dados de treinamento.
    predict(X_test): Retorna as previsões para os dados de teste.
    euclidiana_distance(x1, x2): Calcula a distância euclidiana entre dois pontos.
    
    OBS: 
        Euclidiana_distance é um método auxiliar usado internamente para calcular as distâncias entre os pontos de teste e os pontos de treinamento. 
        Serve para determinar quais são os vizinhos mais próximos.    
    
    calculate_prediction(x): Calcula a previsão para um ponto de teste com base nos vizinhos mais próximos.

Uso:
    1. Instanciar a classe KNN com os parâmetros desejados.
    2. Chamar o método fit() para treinar o modelo com os dados de treinamento.
    3. Chamar o método predict() para obter as previsões para os dados de teste.
    
    Exemplo:
        modelo = KNN(k=3, task='classification')
        modelo.fit(X_train, y_train)
        predictions = modelo.predict(X_test)
"""

# notas
# (a) K-Nearest Neighbors (kNN) – Taís

# * Treinar o classificador utilizando diferentes medidas de distância:
#   • Distância Euclidiana
#   • Distância Manhattan

class KNN:
    def __init__(self, k=5, metric='euclidean', task='classification'):
        self.k = k
        self.metric = metric  # Métrica padrão, pode ser 'euclidean' ou 'manhattan'
        self.task = task


    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        
    """
    euclidiana_distance e manhattan_distance são métodos auxiliares para calcular as distâncias entre os pontos de teste 
    e os pontos de treinamento.
    Eles são usados internamente para determinar quais são os vizinhos mais próximos.
    
    """    
    def _compute_distances(self, X_test):
        """
        Calcula as distâncias entre todos os pontos de teste e todos os pontos de treino
        de forma vetorizada, sem loops Python.
        Retorna uma matriz de shape (n_test, n_train).
        """
        if self.metric == 'euclidean':
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x*y
            X_test_sq  = np.sum(X_test ** 2, axis=1, keepdims=True)
            X_train_sq = np.sum(self.X_train ** 2, axis=1, keepdims=True)
            cross      = X_test @ self.X_train.T
            dists = np.sqrt(np.maximum(X_test_sq + X_train_sq.T - 2 * cross, 0))
        elif self.metric == 'manhattan':
            # Calcula linha por linha para evitar alocação de matriz 3D (n_test, n_train, d)
            dists = np.empty((X_test.shape[0], self.X_train.shape[0]))
            for i, x in enumerate(X_test):
                dists[i] = np.sum(np.abs(self.X_train - x), axis=1)
        else:
            raise ValueError("Metrica deve ser 'euclidean' ou 'manhattan'")
        return dists

    def predict(self, X_test):
        dists     = self._compute_distances(X_test)
        k_indices = np.argsort(dists, axis=1)[:, :self.k]
        k_labels  = self.y_train[k_indices]

        if self.task == 'classification':
            predictions = np.array([
                np.bincount(row.astype(int)).argmax()
                for row in k_labels
            ])
        elif self.task == 'regression':
            predictions = np.mean(k_labels, axis=1)
        else:
            raise ValueError("Task must be either 'classification' or 'regression'")

        return predictions