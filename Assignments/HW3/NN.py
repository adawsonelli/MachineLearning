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
    NeuralNetwork - 3 layer neural network with an input layer, one hidden layer, 
    and a single output node used classification.
    """
    def __init__(self,fileName ="sonar.arff", nFolds = 10, eta = .05, numEpochs = 5000):
        
        #system parameters
        self.nFolds = nFolds
        self.eta = eta
        self.numEpochs = numEpochs
        
        #system hypeParameters
        self.miniBatchSize = 3
        self.ClassThreashold = .5
        
        #initialize dataset:
        self.data = self.importarff(fileName)
        self.pos = self.count('pos')
        self.neg = self.count('neg')
        self.nInstances =  self.data.shape[0]     
        self.nAttributes = self.data.shape[1] - 1 
        
        #separate dataset into nFold stratified Groups:
        self.Folds = self.makeNFolds()
        #self.FoldIDs  #defined elseware in context
        
        #NN state information
        self.W = []  #weights from i to h    [nAttributes x nAttributes]
        self.u = []  #weights from h to y    [nAttributes x 1]'
        self.b = []  #bias vector from       [nAttributes x 1]
        self.c = []  #bias of output y       [1x1]
        
        #vectorized functions
        self.vSigmoid = np.vectorize(sigmoid)
        

    
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
        nInstances =  len(data['data'])
        nAttributes = len(data['data'][0]) - 1   #doesn't include the class label
        self.data = np.zeros([nInstances,nAttributes + 1])
        
        for InstID, instance in enumerate(data['data']):
            for atrID, attribute in enumerate(data['data'][InstID]):
                #handle attributes:
                if type(attribute) == float:
                    self.data[InstID,atrID] = attribute
                #handle class labels:
                elif type(attribute) ==  unicode:
                    self.data[InstID,atrID] = self.CLs[attribute]
        return self.data
                    
    def count(self,label):
        """determine the number of positive and negative instances in the training set"""
        
        if label == 'neg': classValueID = 0
        if label == 'pos': classValueID = 1
      
        count = 0
        for row in self.data:
            if row[-1] == classValueID:
                count += 1
        return count
        
    def makeNFolds(self):
        """
        separate system data into n groups with approximately equal represenation
        of classes. This is known as an n-folds stratified set
        returns: list for numpy arrays
        """
        Folds = []        #init empty fold set
        self.FoldIDs = []      #instance ID's per set will be useful for printing
        
        #make lists of positive and negative instance ID's
        posIDs = [] ; negIDs = []
        for ID,instance in enumerate(self.data):
            if instance[-1] == 0:   #negative
               negIDs.append(ID)
            elif instance[-1] == 1: #pos
                posIDs.append(ID)
        
        #make n fold groups, each are np.arrays
        instancesPerGroup = self.nInstances / float(self.nFolds)  
        
        for grp in range(self.nFolds):
            #determine sampling numbers for this group
            posPercent = self.pos/float(self.nInstances)
            nInstances = intSelect(instancesPerGroup)
            nPos       = intSelect(nInstances * posPercent)
            nNeg       = intSelect(nInstances * (1-posPercent))
            
            #perform sampling
            
            #put together a list of what samples are in this group
            groupSamples = []
            while nPos > 0 and len(posIDs) > 0: #add positives to group:
                popID = np.random.randint(0,len(posIDs))
                sampleID = posIDs.pop(popID)
                groupSamples.append(sampleID)
                nPos -= 1
            while nNeg > 0 and len(negIDs) > 0: #add negatives to group:
                popID = np.random.randint(0,len(negIDs))
                sampleID = negIDs.pop(popID)
                groupSamples.append(sampleID)
                nNeg -= 1
            
            #perform sampling from self.data
            group = np.zeros([len(groupSamples),self.nAttributes])
            for newID,sampleID in enumerate(groupSamples):
                group[newID,:] = self.data[sampleID,:]
                
                
            #add group to folds, instance ID's to fold ID's 
            Folds.append(group)
            self.FoldIDs.append(groupSamples)
        
        return Folds
    
    def nFoldStratifiedCrossValidation(self, fileName):
        """
        This is the top level function that performs a cross validation on the 
        training set and writes the results to the file fileName
        """
        # the process of training the NN - testing the results saving the output
        #must be completed n times - n being the number of folds
        for fID, fold in enumerate(self.Folds):
            folds = self.Folds[:] #make a shallow copy
            testSet = folds.pop(fID)
            trainingSet = merge(folds)
            self.train(trainingSet)
            results = self.test(testSet)
            self.storeTestResults(results)
     
    def train(self, trainingSet):
        """
        train the NN state information - using a stochastic gradient decent 
        approach.
        inputs: 
            training set
        outputs:
            (implicit) state adjustment on W,u,b,c
        """
        #wipe system state clean with new random weights
        self.initState(trainingSet)
        #
        
        
    
    def initState(self):
        """
        initialize the state of the neural network, i.e. all bias vectors and weights
        """
        wScale = .01
        self.W = np.random.normal(0,wScale,(self.nAttributes,self.nAttributes))
        self.u = np.random.normal(0,wScale,(self.nAttributes,1))   #use u'
        self.b = np.random.uniform(0,1,(self.nAttributes,1))  #what should be the scale on this??
        self.c = np.random.uniform(0,1,(1,1))
        
    
    def forwardPropagation(self, x):
        """
        starting with input instance x - determine output y for the current set
        of sytem weights W and u, as well as system biases b and c
        x = input instance       [nAttributes + 1]
        h = hidden layer outputs [nAttribues]        
        """
        #trim class label off of x
        x = x[0:-1]
        
        #perform forward propagation
        h = self.vSigmoid(np.matmul(self.W,x.transpose()) + self.b)
        y = self.vSigmoid(np.matmul(self.u.transpose(),h) + self.c)
        
        return y
    
    def backwardPropagation(self):
        """
        """
        pass
        
    def test(self,testSet):
        """
        """
        pass
    
    def storeTestResults(self,results):
        """
        """
        pass
        
            
            
            
        
        
                
#--------------------------- utility functions --------------------------------

def randBin(prob):
    """
    returns 1 or 0 randomly with probabilty prob
    """
    switch = np.random.uniform(0,1)
    if switch <= prob:
        return 1
    else:
        return 0
    
def intSelect(decimal):
    """
    rounds a decimal into an integer with probability corresponding to it's decimal. 
    for instance:
        7.95 would have a 95 percent chance of becoming 8 
        6.07 would have a 93 percent chance of becoming 6
    """
    integer = int(decimal) ; dec = decimal - integer
    return integer + randBin(dec)
    
def merge(foldList):
    """
    input: list of np.arrays
    output: a single np array with all of the contents of the arrays stacked 
    ontop of one another
    """
    allFolds = np.array([])
    for fold in foldList:
        np.vstack(allFolds,fold)
    return allFolds

def sigmoid(x):
    """
    scalar sigmoid function
    """
    return  1/(1 + np.exp(-x))
        
    
        
        
        
        
        
        
                
        
    
    
    
    
    



