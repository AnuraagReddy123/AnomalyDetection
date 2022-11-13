import numpy as np
from F3_BaseAnomalyClassifier import BaseAnomalyClassifier
from F8_LNFeature import LNFeature

class LNClassifier(BaseAnomalyClassifier):

    def fit(self, X_Train, Y_Train, **kwargs):
        allowed_keys = {'frequency'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
        self.ln = LNFeature()
        self.X_Train_Features_ = self.ln.transform(X_Train, self.frequency)
        self.rf.fit(self.X_Train_Features_, Y_Train)
    
    def predict(self,Test_X, **kwargs):
        allowed_keys = {'frequency'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

        self.X_Test_Features = self.ln.transform(Test_X, self.frequency)
        return self.rf.predict(self.X_Test_Features)
        