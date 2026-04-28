import numpy as np

"""
docstring: 

Classe para pré-processamento de dados, incluindo preenchimento de valores ausentes, 
normalização e divisão em conjuntos de treino e teste.

Atributos:
- X: Matriz de features (n, n_features)
- y: Vetor de target (n,)

Métodos:
- fill_missing: Substitui valores NaN pela média da coluna.
- normalize: Aplica normalização Z-score.
- train_test_split: Divide os dados em conjuntos de treino e teste com uma proporção especificada.

Uso:
1. Instanciar a classe com os dados X e y.
2. Chamar os métodos de pré-processamento em sequência.

Exemplo:
preprocessor = DataPreprocessing(X, y)
preprocessor.fill_missing().normalize()
x_train, x_test, y_train, y_test = preprocessor.train_test_split(test_size=0.2)
"""
class DataPreprocessing:
    
    def __init__(self, X, y):
        self.X = np.array(X, dtype=float)
        self.y = np.array(y, dtype=float)

    def fill_missing(self):
        """Substitui NaN pela média"""
        col_means = np.nanmean(self.X, axis=0)
        inds = np.where(np.isnan(self.X))
        self.X[inds] = np.take(col_means, inds[1])
        return self

    def normalize(self):
        """Normalização Z-score"""
        mean = np.mean(self.X, axis=0)
        std = np.std(self.X, axis=0)
        std[std == 0] = 1
        self.X = (self.X - mean) / std
        return self

    def train_test_split(self, test_size=0.2, seed=42):
        np.random.seed(seed)
        idx = np.random.permutation(len(self.X))
        split = int(len(self.X) * (1 - test_size))
        
        x_train = idx[:split]
        x_test = idx[split:]
        
        return (
            self.X[x_train],
            self.X[x_test],
            self.y[x_train],
            self.y[x_test]
        )
    

