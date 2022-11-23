import numpy as np
from scipy.stats import skew , kurtosis
from F2_BaseFeature import BaseFeature


class PMSFeature(BaseFeature):
    def get_FFT(self,a,k):
        # X_FFT = np.abs((np.fft.rfft(a, axis=1))) Shape: (n_samples, n_features)
        # X_FFT= np.sort(X_FFT, axis=1)[:,::-1] Shape: (n_samples, n_features)
        # return X_FFT[:,:k] Shape: (n_samples, k)

        X_FFT = []
        for wave in a:
            x_fft = np.abs(np.fft.rfft(wave)) # Shape: (n_features,)
            x_fft = np.sort(x_fft)[::-1] # Shape: (n_features,)
            # print(x_fft.shape)
            if (x_fft.shape[0] < k):
                x_fft = np.pad(x_fft, (0, k - x_fft.shape[0]), 'constant', constant_values=(0, 0))
            X_FFT.append(x_fft[:k]) # Shape: (k,)
        
        return np.asarray(X_FFT) # Shape: (n_samples, k)

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
        # X_pow=X**2
        # X_pow= np.sum(X_pow, axis=1) 
        X_pow = []
        for wave in X:
            X_pow.append(np.sum(wave**2))
        X_pow = np.asarray(X_pow)
        features=(np.asarray(X_mean).reshape(-1,1), 
                    np.asarray(X_sd).reshape(-1,1),
                    np.asarray(X_kurtosis).reshape(-1,1),
                    np.asarray(X_skew).reshape(-1,1),
                    X_fft,
                    X_pow.reshape(-1,1))

        # Check for nan values
        for feature in features:
            if np.isnan(feature).any():
                print("Nan values found in feature")
                exit(0)


        return np.concatenate(features,axis=1)