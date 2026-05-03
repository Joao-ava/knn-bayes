import numpy as np


def kfold(X, k_folds):
    """
    Faz a divisão do conjunto de dados k vezes
    A cada K vezes dendo uma parte diferente do conjunto sendo
    o conjunto de teste

    Exemplo
        for train_idx, test_idx in kfold(X, 4):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]
    """
    n_samples = len(X)
  
    idxs = np.arange(n_samples)
    fold_sizes = np.full(k_folds, n_samples // k_folds, dtype=int)
    fold_sizes[:n_samples % k_folds] += 1
  
    current = 0
    results = []
    
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = idxs[start:stop]
        
        train_idx = np.concatenate((idxs[:start], idxs[stop:]))
        results.append((train_idx, test_idx))
        current = stop
        
    return results