import numpy as np
from F3_BaseAnomalyClassifier import BaseAnomalyClassifier
from F11_PMSFeature import PMSFeature

class PMSClassifier(BaseAnomalyClassifier):

    def fit(self,X_Train,Y_Train, **kwargs):
        allowed_keys = {}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
        
        self.pms = PMSFeature()
        self.X_Train_Features_ = self.pms.transform(X_Train)
        self.rf.fit(self.X_Train_Features_,Y_Train)
    
    def predict(self,Test_X, **kwargs):
        allowed_keys = {}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
        
        self.X_Test_Features = self.pms.transform(Test_X)
        return self.rf.predict(self.X_Test_Features)
        