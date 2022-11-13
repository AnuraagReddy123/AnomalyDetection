import numpy as np
from F1_BaseAnomalyGenerator import BaseAnomalyGenerator

class LNAnomalyGenerator(BaseAnomalyGenerator) :
    def transform(self, X, alpha):
        max_standard_deviation = self.max_standard_deviation
        std= np.std(X)*alpha/100
        std=max(std, max_standard_deviation)
        X_new = X + np.random.normal(0,std, X.shape[0])
        return X_new