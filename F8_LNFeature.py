import numpy as np
from F2_BaseFeature import BaseFeature

class LNFeature(BaseFeature):
    def transform(self, X, fs):
        cutoff_param = self.cutoff_param
        F = []
        for i in range(X.shape[0]):
            temp = np.fft.rfft(X[i])          
            p = np.abs(np.fft.rfft(X[i]))     
            f = np.linspace(0, fs/2, len(p))
            cutoff = fs*cutoff_param                  
            index=np.where(f == cutoff)[0][0]  
            temp[:index] = 0                  
            temp1 = np.fft.irfft(temp)       

            F.append(np.sum(temp1**2))
        
        F = np.array(F).reshape(-1,1)
        return F