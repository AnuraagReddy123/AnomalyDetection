import numpy as np
from F1_BaseAnomalyGenerator import BaseAnomalyGenerator

class PMSAnomalyGenerator(BaseAnomalyGenerator) :
    def add_PMS(self,wave,k=10, alpha=1.0, pos=0):
        wave1 = wave.copy()
        seg = wave[pos:pos+k]
        wave1[pos:int(pos+0.1*wave.shape[0])] +=alpha*np.max(seg)
        return wave1

    def transform(self,X, alpha):
        pos= np.random.randint(0,int(0.9*X.shape[0])) 
        X_pms = self.add_PMS(X, int(X.shape[0]*0.1),alpha, pos)

        return X_pms