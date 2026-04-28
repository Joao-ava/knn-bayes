import numpy as np
class NaiveBayes:
    def fit(self, xtrain, ytrain):
        nsamples, nfeatures = xtrain.shape
        self._classes = np.unique(ytrain)
        nclasses = len(self._classes)
        
        self.mean = np.zeros((nclasses,nfeatures),dtype=float)
        self.var = np.zeros((nclasses,nfeatures),dtype=float)
        self.prior = np.zeros(nclasses,dtype=float)
        
xtrain = np.array([[3,2,1],[4,5,6]])
    
nsamples, nfeatures = xtrain.shape
    
print(nsamples)
print(nfeatures)
 
ytrain = ['sim','sim','não','não']

_classes = np.unique(ytrain)
nclasses = len(_classes)
mean = np.zeros((nclasses,nfeatures),dtype=float)

print(f'{mean} erro')

prior = np.zeros(nclasses,dtype=float)

print(f'{prior} erro')
