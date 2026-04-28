import numpy as np
import urllib.request
import math
from fit import KNN
 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

data = np.genfromtxt(urllib.request.urlopen(url),delimiter=",")


# separar x e y
y = data[:,0]
X = data[:,1:]

def train_test_split(X, y, test_size=0.3, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)
    if len(X) != len(y):
        raise ValueError("X e y devem ter o mesmo tamanho")
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    n_test = math.ceil(n_samples * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    if X.ndim == 1:
        X_train, X_test = X[train_indices], X[test_indices]
    else:
        X_train, X_test = X[train_indices,:], X[test_indices,:]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test 


print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y)
modelo = KNN()
pred = modelo.fit(X_train,y_train)
print(modelo.predict(X_test))


y_true = [1,0,1,1,0]
y_pred = [1,0,0,1,0]

def acc(VP ,VN, FP,FN):
    acuracia = (VP + VN) / (VP + FP + FN + VN)
    return acuracia

def precision(VP ,FP):
    return VP / (VP + FP)

def recall(VP ,VN, FP,FN):
    return VP/(VP+FN)  # Da mais valor para falsos negativos
    # o objetivo é diminuir a quantidade de falsos negativos
    # permitindo em certa médida falso positivo
    # exemplo de caso de uso identificar doença
    
def f_score(VP, FP,FN):
    p = precision(VP ,FP)
    r = recall(VP , FN)
    
    return 2* p * r / (p+r)

VP = 2
VN = 2
FP = 1
FN = 0

print(acc(VP ,VN, FP,FN))

y_pred = [1,0,0,1,0]


         
