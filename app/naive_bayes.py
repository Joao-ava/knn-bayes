import numpy as np


"""
docstring:
Implementação do classificador Naive Bayes para classificação univariada.

Atributos:
    _classes (array): Array contendo as classes únicas presentes nos dados de treinamento.
    mean (array): Matriz contendo as médias de cada classe para cada feature.
    var (array): Matriz contendo as variâncias de cada classe para cada feature.
    prior (array): Array contendo as probabilidades a priori de cada classe.
    
    Métodos:
        __init__(self): Inicializa os atributos do classificador.
        fit(self, xtrain, ytrain): Treina o classificador com os dados de treinamento.
        _pdf(self, class_idx, x): Calcula a função de densidade de probabilidade para uma classe e um ponto de dados.
        _predict(self, x): Faz a previsão para um ponto de dados com base nas probabilidades posteriores.
        predict(self, xtest): Retorna as previsões para os dados de teste.
Uso:
    1. Instanciar a classe knn_bayes.
    2. Chamar o método fit() para treinar o modelo com os dados de treinamento.
    3. Chamar o método predict() para obter as previsões para os dados de teste.
    
    Exemplo:
        modelo = knn_bayes()
        modelo.fit(x_train, y_train)
        predictions = modelo.predict(x_test)
"""

# (b) Classificador Bayesiano (Univariado) – Taís

# * Treinar utilizando:
#   • Função de densidade da distribuição normal univariada

class NaiveBayes:  
    def __init__(self):
        self._classes = None
        self.mean = None
        self.var = None
        self.prior = None

    """
    Naive_bayes é um classificador probabilístico baseado no teorema de Bayes, que assume a independência entre as features.
    Nela, o método fit() é responsável por calcular as médias, variâncias e probabilidades a priori para cada classe com base 
    nos dados de treinamento.
    """
    def fit(self, x_train, y_train):
        nsamples, nfeatures = x_train.shape
        self._classes = np.unique(y_train)
        nclasses = len(self._classes)
        self.mean = np.zeros((nclasses,nfeatures),dtype=float)
        self.var = np.zeros((nclasses,nfeatures),dtype=float)
        self.prior = np.zeros(nclasses,dtype=float)
        
        for idx, c in enumerate(self._classes): 
            X_c = x_train[y_train == c] 
            self.mean[idx,:] = X_c.mean(axis=0)
            self.var[idx,:] = X_c.var(axis=0)
            self.prior[idx] = X_c.shape[0]/float(nsamples)
            
    """
    _pdf é um método auxiliar que calcula a função de densidade de probabilidade para uma classe específica e um ponto de dados.
    
    Ele utiliza a fórmula da distribuição normal univariada para calcular a probabilidade de um ponto pertencer a uma classe com base nas 
    médias e variâncias calculadas durante o treinamento.
    
    Serve pra determinar a probabilidade de um ponto de teste pertencer a uma classe específica, o que é essencial para fazer previsões 
    com base no modelo Naive Bayes.
    
    "pdf" significa "probability density function" (função de densidade de probabilidade) e é usada para calcular a probabilidade de um 
    ponto de dados pertencer a uma classe específica com base nas características do modelo treinado.
    """
    def _pdf(self, class_idx, x):
        # pdf - probability density function
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerador = np.exp(-((x-mean)**2) / (2*var))
        denominador = np.sqrt(2*np.pi * var)
        return numerador / denominador
    

    """
    _predict é um método auxiliar que faz a previsão para um ponto de dados específico com base nas probabilidades posteriores 
    calculadas para cada classe.
    
    Método auxiliar pois é usado internamente para calcular a previsão para um único ponto de teste, enquanto o método predict() é 
    responsável por aplicar essa previsão a um conjunto de dados de teste.

    Sua formula calcula as probabilidades posteriores para cada classe usando a probabilidade a priori e a função de densidade de 
    probabilidade, e retorna a classe com a maior probabilidade posterior como a previsão para o ponto de teste.

    Faz log(prior) para evitar problemas de underflow, que podem ocorrer quando se multiplicam muitas probabilidades pequenas, resultando 
    em um número muito próximo de zero.
    """
    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self.prior[idx])  # usa o prior da classe atual
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)
        return self._classes[np.argmax(posterior)]
    
   
    def predict(self, xtest):
        ypred = [self._predict(x) for x in xtest]
        return np.array(ypred)
    