import numpy as np
#visualizacao de dados
import matplotlib.pyplot as plt
import plotly.graph_objects as go
"""fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(x1, x2, y)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
plt.show()
X = np.column_stack((np.ones(len(x1)),x1, x2))
print(X)
X_T = X.T
print("a transposta de X é", X_T)
inversa = np.linalg.inv(X_T @ X)
print("A inversa é", inversa)
beta = np.linalg.inv(X_T @ X) @ X_T @ y
print(beta)
y_pred = X @ beta"""

class MultipleLinearRegression:
    def __init__(self):
        self.beta_hat = None
    def fit(self, X_train, y_train):
        self.N, self.p = X_train.shape
        X_train = np.column_stack((np.ones((self.N,1)), X_train))
        self.beta_hat = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    def predict(self, X_new):
        self.N = X_new.shape[0]
        X_new = np.column_stack((np.ones((self.N,1)), X_new))
        return X_new @ self.beta_hat


        
    
x1 = np.array([2,8,11,10,8,4,2,2,9,8])
x2 = np.array([50, 110, 120, 550, 295, 
               200, 375, 52, 100, 300])
y = np.array([9.95,24.45,31.75,35,25.02,
              16.86,14.38,9.6,24.35,27.5])
