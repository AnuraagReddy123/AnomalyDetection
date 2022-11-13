import numpy as np
from scipy.stats import skew , kurtosis
from F2_BaseFeature import BaseFeature


class PMSFeature(BaseFeature):
    def get_FFT(self,a,k):
        X_FFT = np.abs((np.fft.rfft(a, axis=1)))
        X_FFT= np.sort(X_FFT, axis=1)[:,::-1]
        return X_FFT[:,:k]   

    def transform(self, X):
        n_coeff = self.n_coeff
        X_mean=[]
        X_sd=[]
        X_kurtosis=[]
        X_skew=[]
        for wave in X:    
            temp=np.array(wave)
            X_mean.append(np.mean(temp))        
            X_sd.append(np.std(temp))           
            X_kurtosis.append(kurtosis(temp))  
            X_skew.append(skew(temp))           
        
        
        X_fft=self.get_FFT(X,n_coeff)                     
        X_pow=X**2
        X_pow= np.sum(X_pow, axis=1) 
        features=(np.asarray(X_mean).reshape(-1,1),np.asarray(X_sd).reshape(-1,1),np.asarray(X_kurtosis).reshape(-1,1),np.asarray(X_skew).reshape(-1,1),X_fft,X_pow.reshape(-1,1))
        return np.concatenate(features,axis=1)