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

    def fill_missing(self, method='mean'):
        """Substitui NaN pela média"""
        inds = np.where(np.isnan(self.X))
        if method == 'mean':
            col_means = np.nanmean(self.X, axis=0)
            self.X[inds] = np.take(col_means, inds[1])
        elif method == 'median':
            self.X[inds] = np.take(np.nanmedian(self.X, axis=0), inds[1])
        else:
            self.X[inds] = 0
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
    
    def shuffle(self, seed=42):
        """Retorna X e y em ordem aleatória"""
        np.random.seed(seed)
        idx = np.random.permutation(len(self.X))
        return self.X[idx], self.y[idx]
    
    def select_by_correlation(self, k: int = 1000):
        """Seleciona as k colunas mais correlacionadas com y"""
        # Correlação de Pearson entre cada coluna de X e y, com proteção para variância zero
        y_std = np.std(self.y)
        if y_std == 0 or not np.isfinite(y_std):
            corrs = np.zeros(self.X.shape[1], dtype=float)
        else:
            corrs = np.zeros(self.X.shape[1], dtype=float)
            for j in range(self.X.shape[1]):
                x_col = self.X[:, j]
                x_std = np.std(x_col)
                if x_std == 0 or not np.isfinite(x_std):
                    corrs[j] = 0.0
                else:
                    corr = np.corrcoef(x_col, self.y)[0, 1]
                    corrs[j] = corr if np.isfinite(corr) else 0.0

        corrs = np.nan_to_num(corrs, nan=0.0, posinf=0.0, neginf=0.0)
        # Ordena pelo valor absoluto da correlação
        idx = np.argsort(np.abs(corrs))[::-1][:k]
        new_X = self.X[:, idx]
        print(f'new_x: {new_X.shape}')
        self.X = new_X
        return self

    def balancing_class(self):
        """Trunca cada classe para o tamanho da menor classe"""
        self.y = self.y.astype(int)
        y_counts = np.bincount(self.y)
        min_count = np.min(y_counts)

        selected = []
        for cls in np.unique(self.y):
            idx = np.where(self.y == cls)[0]
            selected.append(idx[:min_count])

        idx_final = np.concatenate(selected)
        np.random.shuffle(idx_final)
        self.X = self.X[idx_final]
        self.y = self.y[idx_final]
        return self
    
    def oversample_minority(self):
        """
        Repete exemplos da classe 1 até igualar a quantidade da classe 0.
        """
        # Índices das classes
        idx_class1 = np.where(self.y == 1)[0]
        idx_class0 = np.where(self.y == 0)[0]

        n_class1 = len(idx_class1)
        n_class0 = len(idx_class0)

        # Quantas vezes precisamos repetir a classe 1
        reps = int(np.ceil(n_class0 / n_class1))

        # Repete os índices da classe 1
        idx_class1_oversampled = np.tile(idx_class1, reps)[:n_class0]

        # Junta os índices
        idx_final = np.concatenate([idx_class0, idx_class1_oversampled])
        np.random.shuffle(idx_final)

        # Novo dataset balanceado
        X_balanced = self.X[idx_final]
        y_balanced = self.y[idx_final].astype(int)  # garante que seja inteiro

        return X_balanced, y_balanced