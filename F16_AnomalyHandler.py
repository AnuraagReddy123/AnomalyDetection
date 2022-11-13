import numpy as np
import pandas as pd
from copy import deepcopy
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
import sys

class MySplitter:           # This class splits the data into train, test sets and stores them in their respective directories.

    def __init__(self, data=None, data_type = None) :
        self.data = data
        self.data_type = data_type

    def split(self, mypath='.', split_size=0.3) :

        train,test = train_test_split(self.data,test_size=split_size)  # Split the data into Train(70%) and Test(30%) sets.
    
        mydf = pd.DataFrame(train)                  # Name the Training set as train
        mydf.to_csv(mypath+'/'+self.data_type+'_train.csv',index=False)
        mydf = pd.DataFrame(test)                   # Name the Test Set as test.
        mydf.to_csv(mypath+'/'+self.data_type+'_test.csv',index=False)


class AnomalyHandler:

    def __init__(self, split_size = 0.3, path_raw_data = None, path_profile_file = None) :   # Constructor reads the normal data
        self.split_size = split_size
        self.path_raw_data = path_raw_data
        self.path_profile_file = path_profile_file
        self.stable_ind = []
        if path_profile_file:
            self.profile=pd.read_csv(self.path_profile_file ,header=None, sep='\t')    # Read the data file profile.txt.
            self.stable_ind = np.where(self.profile[4]==0)[0]

        self.normal_data = []
        if len(self.stable_ind):    # If stable_ind is not empty
            self.normal_data  = pd.read_csv(self.path_raw_data,  header=None, sep='\t').values[self.stable_ind]
        else:
            self.normal_data  = pd.read_csv(self.path_raw_data,  header=None, sep='\t').values


    def addAnomalyGenerator(self, generator): # Add a classifier to the class
        self.generator = generator

    def generateAnomalyData(self, alpha):  # Takes an anomaly generator as input and generates anomaly data
        self.alpha = alpha
        self.data = deepcopy(self.normal_data)
        self.anomaly_data =  self.generator.transform(self.data, alpha = self.alpha)
    
    def getAnomalyData(self):     # Returns anomaly data
        return self.anomaly_data

    def createTrainTestFiles(self, directoryPath) : # This method splits both normal and anomaly data
        self.directoryPath = directoryPath

        if not os.path.exists(self.directoryPath): # If the directory doesnot exist create it.
            os.mkdir(self.directoryPath)

        SN = MySplitter(data_type='normal', data = self.normal_data)    # Splitting Normal data
        SN.split(mypath=self.directoryPath)
        SA = MySplitter(data_type='anomaly', data = self.anomaly_data)  # Splitting Anomaly data
        SA.split(mypath=self.directoryPath)

    def addClassifier(self,classifier) : # Add a classifier to the class
        self.classifier = classifier

    def performClassification(self, directoryPath, **kwargs) :
        self.directoryPath = directoryPath
        self.test_file = self.directoryPath+'/'
        Train_Normal = pd.read_csv(self.directoryPath +'/normal_train.csv').values   # passed directly to fit
        Train_Anomaly = pd.read_csv(self.directoryPath +'/anomaly_train.csv').values  # passed directly to fit

        test_normal= pd.read_csv(self.directoryPath+'/normal_test.csv').values         
        test_anomaly= pd.read_csv(self.directoryPath+'/anomaly_test.csv').values 
        Test_X = np.concatenate((test_normal,test_anomaly))                                         # passed directly to predict
        self.Test_Y = np.concatenate((np.zeros(test_normal.shape[0]), np.ones(test_anomaly.shape[0])))   # To check the accuracy of prediction

        self.classifier.fit(Train_Normal=Train_Normal, Train_Anomaly=Train_Anomaly, **kwargs)
        self.Y_hat = self.classifier.predict(Test_X=Test_X, **kwargs)
    
    def getClassificationOutput(self):
        return self.Y_hat

    def getAccuracy(self):     # Return the accuracy of the model.
        return accuracy_score(self.Test_Y,self.Y_hat)


