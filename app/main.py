
import numpy as np
from figures.regression_plotter import RegressionPlotter
from data import load_fraud_detection
from data_preprocessing import DataPreprocessing

if __name__ == '__main__':

    X, y = load_fraud_detection()
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"X sample:\n{X[:5]}")
    print(f"y sample:\n{y[:5]}")

    pred = DataPreprocessing(X, y)
    pred.fill_missing().normalize()
    x_train, x_test, y_train, y_test = pred.train_test_split(test_size=0.2)
    print("X Treino:\n", x_train[:5])
    print("\nX Teste:\n", x_test[:5])
    print("\ny Treino:\n", y_train[:5])
    print("\ny Teste:\n", y_test[:5])
    