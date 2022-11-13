class BaseAnomalyGenerator :
    def __init__(self, k = 10, max_standard_deviation = 0.05) :

        self.k = k
        self.max_standard_deviation = max_standard_deviation

    # Setters
    def set_k(self,k):
        self.k = k
    
    def set_max_standard_deviation(self,max_standard_deviation):
        self.max_standard_deviation = max_standard_deviation

    # Getters
    def get_k(self):
        return self.k

    def set_max_standard_deviation(self):
        return self.max_standard_deviation

    def transform(self,x) :
        pass