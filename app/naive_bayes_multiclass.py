import numpy as np

class NaiveBayesMultiClass:
    def __init__(self):
        self._classes = None
        self.means = None
        self.cov = None
        self.prior = None
    
    def fit(self, X_train, y_train):
        self._classes = np.unique(y_train)
        nclasses = len(self._classes)
        nsamples, nfeatures = X_train.shape
        
        self.means = np.zeros((nclasses, nfeatures), dtype=float)
        self.cov = np.zeros((nclasses, nfeatures, nfeatures), dtype=float)
        self.prior = np.zeros(nclasses, dtype=float)
        
        for idx, c in enumerate(self._classes):
            X_c = X_train[y_train == c]
            self.means[idx, :] = X_c.mean(axis=0) # [idx, :] Linha idx, para todas as colunas
            self.cov[idx, :] = np.cov(X_c, rowvar=False) + 1e-6*np.eye(nfeatures)
            self.prior[idx] = X_c.shape[0] / float(nsamples)
    
    def _log_pdf(self, class_idx, X):
        # Calcula diretamente o log da densidade de probabilidade (Log-PDF)
        d = len(self.means[class_idx])
        det_cov = np.linalg.det(self.cov[class_idx])
        inv_cov = np.linalg.inv(self.cov[class_idx])
        
        # Log da constante de normalização: -0.5 * (d*log(2*pi) + log(det_cov))
        log_norm_const = -0.5 * (d * np.log(2 * np.pi) + np.log(det_cov))
        diff = X - self.means[class_idx]
        
        # O termo do expoente já é linearizado pelo logaritmo
        log_exponent = -0.5 * (diff.T @ inv_cov @ diff)
        
        return log_norm_const + log_exponent
    
    def _predict(self, X):
        posteriors = []
        for idx, c in enumerate(self._classes):
            log_prior = np.log(self.prior[idx]) 
            log_likelihood = self._log_pdf(idx, X) # Recebe diretamente o log
            
            # Soma log(prior) + log(likelihood)
            posterior = log_prior + log_likelihood
            posteriors.append(posterior)
            
        # Retorna a classe usando a lista inteira de posteriors
        return self._classes[np.argmax(posteriors)]
    
    def predict(self, Xtest):
        ypred = [self._predict(x) for x in Xtest]
        return np.array(ypred)