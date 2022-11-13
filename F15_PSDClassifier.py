import numpy as np
from F3_BaseAnomalyClassifier import BaseAnomalyClassifier
from F14_PSDFeature import PSDFeatures

class PSDClassifier(BaseAnomalyClassifier):

    def fit(self,X_Train, Y_Train, **kwargs):
        allowed_keys = {'kernel_size', 'sliding_window_size', 'filter_type'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

        self.psd = PSDFeatures()
        self.X_Train_Features_ = self.psd.transform(X_Train,self.filter_type, self.kernel_size,self.sliding_window_size)

        self.X_Train_Features_ = np.array(self.X_Train_Features_).reshape(-1,1)
        self.rf.fit(self.X_Train_Features_,Y_Train)
    
    def predict(self,Test_X, **kwargs):

        allowed_keys = {'kernel_size', 'sliding_window_size', 'filter_type'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

        self.X_Test_Features = self.psd.transform(Test_X, self.filter_type, self.kernel_size, self.sliding_window_size)
        self.X_Test_Features = np.array(self.X_Test_Features).reshape(-1,1)
        return self.rf.predict(self.X_Test_Features)