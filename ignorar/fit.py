import numpy as np
# openml
# kaggle
# uc irvine

class KNN():
    def __init__(self, k=5, task="classification"):
        self.k = k
        self.task = task
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def euclidian_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
    def calculate_prediction(self, x):
        distances = [self.euclidian_distance(x, x_train) 
                     for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        if self.task == "classification": #voto majoritário
            unique, counts = np.unique(k_nearest_labels, 
                                       return_counts=True)
            return unique[np.argmax(counts)]
        elif self.task=="regression": #media
            return np.mean(k_nearest_labels)
        else:  
            raise ValueError("Task must be either ´classification´ or ´regression´")       
    def predict(self, X_test):
            predictions = [self.calculate_prediction(x) for x in X_test]
            return np.array(predictions)

modelo = KNN()  

X_train = np.array([[1,2],
           [2,3],
           [3,3],
           [10,10]])
y_train=np.array([0,0,1,1])
x = np.array([2,2] )


modelo.fit(X_train, y_train)   # armazenar os dados
print(modelo.predict(x))   # preditivo
def euclidian_distance(x1, x2):
    x1=np.array(x1)
    x2=np.array(x2)
    return np.sqrt(np.sum((x1-x2)**2))
distances = [euclidian_distance(x, x_train) 
             for x_train in X_train]
print(distances)
k=3
k_indices = np.argsort(distances)[:k]    
print(k_indices)

lista = ["maca", "laranja", "maca", "maca"]
unique, counts = np.unique(lista, return_counts=True)
print("Elementos unicos", unique)
print("Contagem", counts)
print(unique[np.argmax(counts)])


