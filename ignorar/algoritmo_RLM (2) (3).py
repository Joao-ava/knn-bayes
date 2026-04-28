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
class RegressionLinearM:
    def __init__(self, X,y): #construtor
        self.X = X
        self.y = y
        self.beta = None #parametros
        self.N = X.shape[0]
    def fit(self): #treinamento
        self.X = np.column_stack((np.ones((self.N)), self.X))
        
    
x1 = np.array([2,8,11,10,8,4,2,2,9,8])
x2 = np.array([50, 110, 120, 550, 295, 
               200, 375, 52, 100, 300])
y = np.array([9.95,24.45,31.75,35,25.02,
              16.86,14.38,9.6,24.35,27.5])
