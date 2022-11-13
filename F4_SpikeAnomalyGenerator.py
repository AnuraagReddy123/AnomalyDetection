import numpy as np
from F1_BaseAnomalyGenerator import BaseAnomalyGenerator


class SpikeAnomalyGenerator(BaseAnomalyGenerator):
    def add_spike(self, wave, k, alpha, pos=0):
        wave1 = wave.copy()
        seg = wave[pos:pos + k]
        wave1[pos] = wave1[pos] + alpha * np.max(seg)
        return wave1

    def transform(self, X, alpha):
        k = self.k
        pos = np.random.randint(0, (X.shape[0] - 10))  # change through formula
        X_ss = self.add_spike(X, k=k, alpha=alpha, pos=pos)
        return X_ss
