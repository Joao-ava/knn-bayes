import numpy as np
from data import load_fraud_detection
from data_preprocessing import DataPreprocessing
from knn import KNN

if __name__ == '__main__':

    X, y = load_fraud_detection()
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

    modelo = KNN()
    modelo.fit(x_train, y_train)
    predictions = modelo.predict(x_test)
    print("\nPredições:\n", predictions[:20])
    print("\nValores Reais:\n", y_test[:20])

    # modelo.fit_bayes(x_train, y_train)
    # predictions_bayes = modelo.predict_bayes(x_test)
    # print("\nPredições Naive Bayes:\n", predictions_bayes[:20])

