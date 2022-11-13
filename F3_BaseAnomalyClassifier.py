from sklearn.ensemble import RandomForestClassifier

class BaseAnomalyClassifier:
    def __init__(self):
        self.rf = RandomForestClassifier(random_state=42)

    def fit(self,Train_X,Train_Y):
        self.rf.fit(Train_X,Train_Y)

    def predict(self,Test_X):
        self.Y = self.rf.predict(Test_X)
        return self.Y