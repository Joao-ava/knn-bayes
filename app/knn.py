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

class KNN():


    def __init__(self,k=5, metric='euclidean'):
        self.k = k
        self.metric = metric  # Métrica padrão, pode ser 'euclidean' ou 'manhattan'


    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        
    """
    euclidiana_distance e manhattan_distance são métodos auxiliares para calcular as distâncias entre os pontos de teste 
    e os pontos de treinamento.
    Eles são usados internamente para determinar quais são os vizinhos mais próximos.
    
    """    
    def euclidiana_distance(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    """
    manhattan_distance é um método auxiliar para calcular a distância Manhattan entre dois pontos.
    Ele é usado internamente para determinar quais são os vizinhos mais próximos quando a métrica escolhida é 'manhattan'.
    """
    def manhattan_distance(self,x1,x2):
        return np.sum(np.abs(x1-x2))

    """
    calculate_prediction é um método auxiliar para calcular a previsão para um ponto de teste com base nos vizinhos mais próximos.
    Ele é usado internamente pelo método predict() para obter as previsões para os dados de teste.
    condicionalmente, ele calcula as distâncias usando a métrica escolhida (euclidiana ou manhattan) e retorna 
    a previsão com base na maioria dos vizinhos para classificação ou na média para regressão.   
    """
    def calculate_prediction(self, x):
        if self.metric == 'euclidean':
            distances = [self.euclidiana_distance(x,x_train) 
                         for x_train in self.X_train]
        elif self.metric == 'manhattan':
            distances = [self.manhattan_distance(x,x_train) 
                         for x_train in self.X_train]
        else:
            raise ValueError("Métrica deve ser 'euclidean' ou 'manhattan'")
        
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        
        unique, counts = np.unique(k_nearest_labels,
                                  return_counts=True)
        return unique[np.argmax(counts)]
     
    
    def predict(self,X_test):
        predictions = [self.calculate_prediction(x) for x in X_test]
        return np.array(predictions)
    

