import numpy as np
from scipy.signal import find_peaks
from F2_BaseFeature import BaseFeature


class SpikeFeature(BaseFeature):
    # def __init__(self, prominence):
    #     self.prominence  = prominence

    def transform(self, X):
        prominence = self.prominence
        F = []
        for i in range(X.shape[0]):
            index = find_peaks(X[i], prominence=prominence)[0]
            index = index[(index > 0) & (index < (X[i].shape[0] - 1))]
            index_left = index - 1
            index_right = index + 1
            temp_left = X[i][index_left]
            temp_right = X[i][index_right]
            temp_mean = temp_left + temp_right
            temp_mean = temp_mean / 2
            original = X[i][index]
            resultant = original - temp_mean
            if resultant.size == 0:
                F.append(0)
            else:
                result = np.max(resultant)
                F.append(result)

        F = np.array(F).reshape(-1, 1)
        means = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            means[i] = np.mean(X[i])
        means = means.reshape(-1, 1)
        F = np.concatenate((F, means), axis=1)

        return F
