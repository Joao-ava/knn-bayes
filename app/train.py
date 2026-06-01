import time
import json
import numpy as np
from pprint import pprint
from collections import defaultdict
from data import load_fraud_detection, load_santander, load_melbourne_housing
from regression import MultiLinearRegression
from kfold import kfold
from knn import KNN
from metrics import accuracy, precision, recall, f1_score, r2_score, r2_score_adjusted, rss, rmse, mae
from data_preprocessing import DataPreprocessing
from naive_bayes import NaiveBayes, NaiveBayesMultiClass
from nn import MLP, sigmoid, binary_cross_entropy, r2_loss, mse
from perceptron import Perceptron


def make_classification_mlp():
    return MLP(
        input_size=28, hidden_sizes=[64, 32], output_size=1,
        loss_function=binary_cross_entropy, output_activation=sigmoid,
        epochs=80, lr=0.01, decay=0.8,
        tolerance=1e-5, patience=4,
        clip_grad=True, batch_size=256
    )


classification_models = [
    ('knn_euclidean', lambda : KNN(5, 'euclidean')),
    ('knn_manhattan', lambda : KNN(5, 'manhattan')),
    ('naive_bayes_univariado', lambda : NaiveBayes()),
    ('naive_bayes_multivariado', lambda : NaiveBayesMultiClass()),
    ('perceptron', lambda : Perceptron()),
    ('mlp_classification', make_classification_mlp),
]
classification_metrics = [accuracy, precision, recall, f1_score]

def make_regression_mlp():
    return MLP(
        input_size=10, hidden_sizes=[64, 32], output_size=1,
        loss_function=mse,
        epochs=60, lr=0.01, decay=0.7,
        tolerance=1e-5, patience=4,
        clip_grad=True, batch_size=128
    )

regression_models = [
    ('knn_euclidean', lambda : KNN(25, 'euclidean', 'regression')),
    ('knn_manhattan', lambda : KNN(25, 'manhattan', 'regression')),
    ('linear_regression', lambda : MultiLinearRegression()),
    ('mlp_regression', make_regression_mlp),
]
regression_metrics = [r2_score, r2_score_adjusted, rss, rmse, mae]

def train_classification_model(folds: int):
    print('Começando treinamento de classificadores')
    model_metrics = defaultdict(dict)
    X, y = load_fraud_detection()
    preprocessing = DataPreprocessing(X, y)
    X, y = preprocessing.fill_missing().normalize().shuffle()

    for model_name, Model in classification_models:
        metrics = defaultdict(list)
        for train_idx, test_idx in kfold(X, folds):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            X_bal, y_bal = DataPreprocessing(X_train, y_train).oversample_minority()
            if model_name != 'mlp_classification':
                X_bal, y_bal = X_train, y_train

            model = Model()

            start_time_fit = time.time()
            model.fit(X_bal, y_bal)
            time_fit = time.time() - start_time_fit
            metrics['time_fit'].append(time_fit)

            start_time_pred = time.time()
            y_pred_proba = model.predict(X_test).reshape(-1)
            y_pred = (y_pred_proba >= 0.5).astype(int)
            time_pred = time.time() - start_time_pred
            metrics['time_pred'].append(time_pred)

            for metric in classification_metrics:
                metric_name = metric.__name__
                metrics[metric_name].append(metric(y_test, y_pred))


        model_metrics[model_name]['mean-time_fit'] = np.mean(metrics['time_fit'])
        model_metrics[model_name]['std-time_fit'] = np.std(metrics['time_fit'])
        model_metrics[model_name]['mean-time_pred'] = np.mean(metrics['time_pred'])
        model_metrics[model_name]['std-time_pred'] = np.std(metrics['time_pred'])
        for metric in classification_metrics:
            metric_name = metric.__name__
            model_metrics[model_name][f'mean-{metric_name}'] = np.mean(metrics[metric_name])
            model_metrics[model_name][f'std-{metric_name}'] = np.std(metrics[metric_name])

    pprint(model_metrics)
    with open('classification_table.json', 'w') as f:
        f.write(json.dumps(model_metrics, indent=2))


def train_regressor_model(folds: int):
    print('Começando treinamento do regressor')
    model_metrics = defaultdict(dict)
    # X, y = load_santander()
    X, y = load_melbourne_housing()
    preprocessing = DataPreprocessing(X, y)
    # X, y = preprocessing.fill_missing().normalize().select_by_correlation(100).shuffle()
    X, y = preprocessing.fill_missing().minmax_scale().normalize().shuffle()
    print(X.shape)
    _, columns_size = X.shape

    for model_name, Model in regression_models:
        metrics = defaultdict(list)
        print(model_name)
        for train_idx, test_idx in kfold(X, folds):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            model = Model()
            y_mean = np.mean(y_train)
            y_std = np.std(y_train)
            if y_std == 0:
                y_std = 1
            y_train_scaled = (y_train - y_mean) / y_std

            start_time_fit = time.time()
            model.fit(X_train, y_train_scaled)
            time_fit = time.time() - start_time_fit
            metrics['time_fit'].append(time_fit)

            start_time_pred = time.time()
            y_pred_scaled = model.predict(X_test).reshape(-1)
            y_pred = y_pred_scaled * y_std + y_mean
            time_pred = time.time() - start_time_pred
            metrics['time_pred'].append(time_pred)

            metrics['r2_score'].append(r2_score(y_test, y_pred))
            metrics['r2_score_adjusted'].append(r2_score_adjusted(y_test, y_pred, p=columns_size))
            metrics['rss'].append(rss(y_test, y_pred))
            metrics['rmse'].append(rmse(y_test, y_pred))
            metrics['mae'].append(mae(y_test, y_pred))


        model_metrics[model_name]['mean-time_fit'] = np.mean(metrics['time_fit'])
        model_metrics[model_name]['std-time_fit'] = np.std(metrics['time_fit'])
        model_metrics[model_name]['mean-time_pred'] = np.mean(metrics['time_pred'])
        model_metrics[model_name]['std-time_pred'] = np.std(metrics['time_pred'])
        for metric in regression_metrics:
            metric_name = metric.__name__
            model_metrics[model_name][f'mean-{metric_name}'] = np.mean(metrics[metric_name])
            model_metrics[model_name][f'std-{metric_name}'] = np.std(metrics[metric_name])

    pprint(model_metrics)
    with open('regression_table.json', 'w') as f:
        f.write(json.dumps(model_metrics, indent=2))


if __name__ == '__main__':
    k = 5
    train_classification_model(k)
    train_regressor_model(k)