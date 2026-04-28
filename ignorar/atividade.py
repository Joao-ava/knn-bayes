import math
import numpy as np


df = np.loadtxt('mt_cars - mt_cars (2).csv',
                delimiter=',', skiprows=1)
# print(df)
# ai ela ta falando sobre baixar o conjunto
# mas pelo que entendi depois vai ser KNN
# Tem que dividir o conjunto de dados tambem
X = df[:, 1:10] # caracteristicas de entrada
y = df[:, 0] # consumo
# O que é hould out???????????

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MultipleLinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Real", y_test)
print("Previstos", y_pred)

def r2_score(y_true, y_pred):
    rss = np.sum((y_true - y_pred) ** 2)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (rss / tss)
    return r2


print(r2_score(y_test, y_pred))


# Multiclasses - muitas classes

acertos = 0

for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        acertos+=1
        
acc = acertos/len(y_test)

precisoes = []

código , pq deu isso, algoritimo, iinterrogatório? dia 4 limite, se não zero, slides fazer

import time tempo de resposta do modelo
 

