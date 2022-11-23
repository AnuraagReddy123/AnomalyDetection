import Constants

class BaseFeature:
    def __init__(self, prominence=0.01, cutoff_param=0.4, n_coeff=Constants.N_COEFF):
        self.prominence = prominence
        self.cutoff_param = cutoff_param
        self.n_coeff = n_coeff

    # Setters
    def set_prominence(self, prominence):
        self.prominence = prominence

    def set_cutoff_param(self, cutoff_param):
        self.cutoff_param = cutoff_param

    def set_n_coeff(self, n_coeff):
        self.n_coeff = n_coeff

    # Getters
    def get_prominence(self):
        return self.prominence

    def get_cutoff_param(self):
        return self.cutoff_param

    def get_n_coeff(self):
        return self.n_coeff

    def transform(self, X):
        pass
