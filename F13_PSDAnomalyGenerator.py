import numpy as np
from F1_BaseAnomalyGenerator import BaseAnomalyGenerator

class PSDAnomalyGenerator(BaseAnomalyGenerator) :
    def add_PSD(self, wave, k, alpha, pos):
        wave1= wave.copy()
        wave1[pos:int(pos+k)]+= np.random.normal(0,alpha,k)
        return wave1

    def transform(self, X, alpha):
        pos= np.random.randint(0,int(0.9*X.shape[0]))
        X_psd = self.add_PSD(X,int(0.1*X.shape[0]),alpha, pos)
        return X_psd