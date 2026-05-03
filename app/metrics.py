import numpy as np

"""
- [ ] Accuracy
- [ ] F1-score
- [ ] Precision
- [ ] Recall
- [ ] R² score
- [ ] R² score ajustado
"""

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred)


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    total = true_positive + false_positive
    if total <= 0: # evitar divisão por zero
        return 0

    return true_positive / total


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    total = true_positive + false_negative
    if total <= 0: # evitar divisão por zero
        return 0

    return true_positive / total


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    total = p + r
    if total <= 0: # evitar divisão por zero
        return 0

    return 2 * p * r / total


def r2_score(y: np.ndarray, y_pred: np.ndarray):
    y_mean = np.mean(y)
    total = np.sum((y - y_mean) ** 2)
    if total <= 0: # evitar divisão por zero
        return 0

    value = np.sum((y - y_pred) ** 2) / total
    return 1 - (value)


def r2_score_adjusted(y: np.ndarray, y_pred: np.ndarray, p: int = 4991):
    n = len(y)
    total = (n - p - 1)
    if total <= 0: # evitar divisão por zero
        return 0

    value = (n - 1) / total
    return 1 - (1 - r2_score(y, y_pred)) * value
