import numpy as np
from scipy.signal import savgol_filter
from F2_BaseFeature import BaseFeature

class PSDFeatures(BaseFeature):
    def getNoise(self, X, filter_type, kernel_size):
        X_filter=[]
        X_noise=[]

        if filter_type =='median':         
            for i in range(X.shape[0]):    
                temp = np.lib.stride_tricks.sliding_window_view(X[i],kernel_size)   
                temp= np.median(temp, axis=1)   
                X_filter.append(temp)           
                X_noise.append(X[i][kernel_size//2:-kernel_size//2+1]-temp) 

        if filter_type == 'savgol' :       
            for i in range(X.shape[0]):
                temp =savgol_filter(X[i],kernel_size, polyorder=1) 
                X_filter.append(temp)           
                X_noise.append(X[i]-temp)       

        X_filter= np.array(X_filter)
        X_noise= np.array(X_noise)

        return X_noise

    def maxMovingEnergy(self, wave, sliding_window_size):
        temp =np.copy(wave)        
        temp_split=np.lib.stride_tricks.sliding_window_view(temp,sliding_window_size) 
        temp_1= temp_split**2               
        temp_2= np.sum(temp_1, axis=1)[1:]  
        return np.max(temp_2)

    def transform(self, X, filter_type, kernel_size, sliding_window_size):
        X_Noise = self.getNoise(X,filter_type,kernel_size)
        F=[]
        for i in range(X.shape[0]):
            F.append(self.maxMovingEnergy(X_Noise[i],sliding_window_size))
        
        return F