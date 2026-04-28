import numpy as np
import math
"""x = np.array([34,5,6,4])
random_state = None
np.random.seed(random_state)
print(np.random.permutation(x)) #embaralhar
tes_size = 0.3
n_samples = len(x)
n_test = math.ceil(n_samples * tes_size) #arredonda para cima
print(n_test)"""
def train_test_split(X, y, test_size=0.3, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)
    if len(X) != len(y):
        raise ValueError("X e y devem ter o mesmo tamanho")
    n_samples = len(X)
    print("Quantidade da amostra", n_samples)
    indices = np.random.permutation(n_samples)
    print("Indices embaralhados", indices)
    n_test = math.ceil(n_samples * test_size)
    print("Quantidade de amostras para teste", n_test)
    test_indices = indices[:n_test]
    print("Dados de teste", test_indices)
    train_indices = indices[n_test:]
    print("Dados de treino", train_indices)
    if X.ndim == 1:
        X_train, X_test = X[train_indices], X[test_indices]
    else:
        X_train, X_test = X[train_indices,:], X[test_indices,:]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test 

 
x1 = np.array([2,8,11,10,8,4,2,2,9,8])
x2 = np.array([50, 110, 120, 550, 295, 
               200, 375, 52, 100, 300])
y = np.array([9.95,24.45,31.75,35,25.02,
              16.86,14.38,9.6,24.35,27.5])
X = np.column_stack((x1, x2))
print("Numero de dimensoes", X.ndim)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
    
# accuracia: acerto, diagonal principal
# problema da acurácia, acertou muito, pode ser um modelo desbalanciado.
# problema binario

    

#[8 1 5 0 7 2 9 4 3 6]-> random_state=42