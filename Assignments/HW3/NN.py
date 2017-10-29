"""
file:NN.py  - neural network library for training and predicting a 3 layer NN
based on an input WEKA file.
author:Alex Dawson-Elli
class:CS 760 Machine learning
"""




#--------------------------import statements ----------------------------------
import arff
import numpy as np
from matplotlib import pyplot as plt




#------------------------------- classes --------------------------------------
class NeuralNetwork:
    """
    neuralnet trainfile num_folds learning_rate num_epochs 
    """
    def __init__(self,fileName,nFolds = 10,eta = .05, numEpochs = 5000):
        
        # system parameters
        self.nFolds = nFolds
        self.eta = eta
        self.numEpochs = numEpochs
        
        #initialize dataset:
        self.importarff(fileName)
        
        #separate dataset into nFold stratified Groups:
        self.Folds = self.makeNGroups(self)
       
        
    
    def importarff(self,fileName):
        """
        processes an input .arff file into a system matrix of the following format:
            np.array -   atr1   atr2    atr3   classLabel
            instance1  [ d11    d12    d13 ...
            instance2    d21    d22    d23 ...
            instance3    d31    d32    d33 ...]
        
        all atr values are real numbers, all class labels are 0 or 1
        """
        #open data file
        data = arff.load(open(fileName,'rb'))
        
        #extract class labels from arff data structure
        CLs = data['attributes'][-1][1]
        self.CLs = {CLs[0]:0, CLs[1]:1}  #lookup dict
        
        #copy all data instances into an np array, including 0 and 1 for class labels
        self.nInstances =  len(data['data'])
        self.nAttributes = len(data['data'][0]) - 1   #doesn't include the class label
        self.data = np.zeros([self.nInstances,self.nAttributes + 1])
        
        for InstID, instance in enumerate(data['data']):
            for atrID, attribute in enumerate(data['data'][InstID]):
                #handle attributes:
                if type(attribute) == float:
                    self.data[InstID,atrID] = attribute
                #handle class labels:
                elif type(attribute) ==  unicode:
                    self.data[InstID,atrID] = self.CLs[attribute]
        
    def makeNGroups(self):
        """
        """
        
        
                
        
    
    
    
    
    



