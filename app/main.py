import numpy as np
import time
from data import load_fraud_detection
from data_preprocessing import DataPreprocessing
from knn import KNN
from naive_bayes_multiclass import NaiveBayesMultiClass
from kfold import kfold

if __name__ == '__main__':

    X, y = load_fraud_detection()
    
    
    for X_train_idx, X_test_idx in kfold(X, 4):
        X_train = X[X_train_idx]
        y_train = y[X_train_idx]
        model = KNN()
        model.fit(X_train, y_train)

        X_test = X[X_test_idx]
        y_pred = model.predict(X_test)
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"X sample:\n{X[:5]}")
    print(f"y sample:\n{y[:5]}")

    pred = DataPreprocessing(X, y)
    pred.fill_missing().normalize()
    x_train, x_test, y_train, y_test = pred.train_test_split(test_size=0.2)

    print("X Treino:\n", x_train[:20])
    print("\nX Teste:\n", x_test[:20])
    print("\ny Treino:\n", y_train[:20])
    print("\ny Teste:\n", y_test[:20])

    # KNN
    modelo = KNN()
    modelo.fit(x_train, y_train)
    start_time = time.time()
    predictions = modelo.predict(x_test)
    knn_time = time.time() - start_time

    print(f"\nKNN - Tempo de predição: {knn_time:.4f} segundos")

    print("\nPredições:\n", predictions[:20])
    print("\nValores Reais:\n", y_test[:20])

    # Naive Bayes
    modelo = NaiveBayesMultiClass()
    modelo.fit(x_train, y_train)
    start_time = time.time()
    predictions = modelo.predict(x_test)
    nb_time = time.time() - start_time

    print(f"\nNaive Bayes - Tempo de predição: {nb_time:.4f} segundos")

    print("\nPredições:\n", predictions[:20])
    print("\nValores Reais:\n", y_test[:20])