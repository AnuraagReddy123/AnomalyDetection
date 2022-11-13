import numpy as np
from F3_BaseAnomalyClassifier import BaseAnomalyClassifier
from F5_SpikeFeature import SpikeFeature

class SpikeClassifier(BaseAnomalyClassifier):
   
    def fit(self, X_Train, Y_Train, **kwargs):
        allowed_keys = {}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
        self.sp = SpikeFeature()
        self.X_Train_Features_ = self.sp.transform(X_Train)
        self.rf.fit(self.X_Train_Features_,Y_Train)

    def predict(self,Test_X, **kwargs):
        allowed_keys = {}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
        
        self.X_Test_Features = self.sp.transform(Test_X)
        return self.rf.predict(self.X_Test_Features)
        