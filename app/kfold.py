import numpy as np

def kfold(X, k_folds):
    n_samples = len(X)
  
    idxs = np.arange(n_samples)
    fold_sizes = np.full(k_folds, n_samples // k_folds, dtype=int)
    fold_sizes[:n_samples % k_folds] =+ 1
  
    current = 0
    results = []
    
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = idxs[start:stop]
        
        train_idx = np.concatenate((idxs[:start], idxs[stop:]))
        results.append((train_idx, test_idx))
        current = stop
        
    return results