import numpy as np
class KNN():
    def __init__(self,k=5, task='classification'):
        self.k = k
        self.task = task
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def euclidiana_distance(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    def calculate_prediction(self, x):
        distances = [self.euclidiana_distance(x,x_train) 
                     for x_train in self.X_train]
        
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        if self.task == "classification":
            unique, counts = np.unique(k_nearest_labels,
                                      return_counts=True)
            return unique[np.argmax(counts)]
        elif self.task == "regression":
            return np.mean(k_nearest_labels)
        else:
            raise ValueError("Task must be either ´classification´ or ´regression´")
    
    def predict(self,X_test):
        predictions = [self.calculate_prediction(x) for x in X_test]
        return np.array(predictions)
    
modelo = KNN()
modelo.fit()
modelo.predict()

        