import numpy as np


class MultiLinearRegression:
    """Classe para regressão linear com x sendo uma matriz"""
    def __init__(self):
        self.b = None

    def fit(self, x, y):
        data = np.column_stack((np.ones(len(y)), x))
        self.b = np.linalg.pinv(data.T @ data) @ (data.T @ y)
        return self.predict(x)


    def predict(self, x):
        data = np.column_stack((np.ones(len(x)), x))
        return data @ self.b
