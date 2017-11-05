"""
file:BNT.py  implementation of naive Bayes aNd TAN (tree-augmented naive Bayes)
author:Alex Dawson-Elli
class:CS 760 Machine learning
"""
#--------------------------import statements ----------------------------------
import arff
import numpy as np
from matplotlib import pyplot as plt


#--------------------------- classes ------------------------------------------

class NaiveBayes():
    """
    implementation of a naiveBayes binary classifier
    """
    def __init__(self, fileName):
        """
        inputs: fileName of training set data file
        """
        #process data  input file:
        self.data , self.atr = importarff(fileName)
        self.nInstances = self.data.shape[0]
        self.nAttributes = self.data.shape[1] -1 #doesn't include CL
        self.nClasses = len(self.atr['names'][-1])    #should always be 2
        
        #train probabilities:                                         #http://www.saedsayad.com/naive_bayesian.htm
        self.classInstances , self.classProbs = self.train("class")   #Class Prior Probability and instances
        self.conditionalProbs = self.train("conditionalProbs")        #conditional probabilities
        self.atrProbs = self.train("PredictorPrior")                             #predictor prior probabilities
    
    def train(self,Tset):
        """
        train a naive bayes classifier - both class probabilites and atrProbs
        structure: 
            self.classProbs     = np.array[1,2]
            self.classInstances = np.array[1,2]
            self.class 
        """
        # train class probabilities: P(O = T) and P(O = F)
        if Tset == "class":  #assumes binary classification problem
            
            #count class instances:
            classInstances = np.zeros([1,2])  # [posCount ,negCount]
            for instance in self.data: 
                if instance[-1] == 0: #pos
                    classInstances[0,0] += 1
                elif instance[-1] == 1: #neg
                    classInstances[0,1] += 1
            
            # calculate class probabilities (using laplace correction)
            classProbs = np.zeros([1,2])
            for ID, CI in enumerate(classInstances):
                classProbs[0,ID] = float(classInstances[0,ID] + 1) / (self.nInstances + self.nClasses)
                
            return classInstances, classProbs
        
        #train attribute conditional probabilites
        #output is:  self.conditionalProbs[hypID][atrID][choiceID] == dim([2][nAttributes][nChoices])
        elif Tset == "conditionalProbs":
            
            hypProbs = []  # [1x2]
            for outputClass in range(self.nClasses):  #binary classification for now
                #calculate counts(A_n and Y):
                P_AnY = []
                for atrID, attribute in enumerate(self.atr['DC']):
                    nChoices = len(attribute)
                    atrProbs = []
                    for  choice in attribute:
                        count = 0
                        #loop over data and count instances:
                        for instance in self.data:
                            if (instance[atrID] == choice) and instance[-1] == outputClass:
                                count += 1
                        #calculate attribute probability:
                        P = float(count + 1) / (self.classInstances[0,outputClass] + nChoices )
                        #assemble lists:
                        atrProbs.append(P)
                    P_AnY.append(atrProbs)
                hypProbs.append(P_AnY)
            return hypProbs
        
        #train predictor prior probabilities
        #output is self.atrProbs[atrID][choiceID]
        elif Tset == "PredictorPrior":
            Ppriors = []
            for atrID, attribute in enumerate(self.atr['DC']):
                nChoices = len(attribute)
                atrProbs = []
                for choice in attribute:
                    count = 0
                    #loop over data and count instances
                    for instance in self.data:
                        if (instance[atrID] == choice):
                            count += 1
                    #calculate attribute probability:
                    P = float(count + 1) / (self.nInstances + nChoices )
                    #assemble lists:
                    atrProbs.append(P)
                Ppriors.append(atrProbs)
            return Ppriors
            
            
            
        
    def predictInstance(self,x):
        """
        use trained naive Bayes binary classifier to predict the class of an input
        instance
        input: 1d np.array vector
        outputs: predictedClass, probability
        """
        #trim off class label:
        x = x[0:-1]
        
        #generated predicted probabilities for each class:
        posteriors = []
        outputClasses = [0,1]
        for hyp in outputClasses: #select h
            P = 1
            #p(x | h) = p(d1 | h) * p(d_2 | h ) * ... p(d_n | h )
            for atrID,choice in enumerate(x):
                P = P*self.atrProbs[hyp][atrID][int(choice)]
            #factor in P(h)
            P = P*self.classProbs[0,hyp]
            
            #add to list
            posteriors.append(P)
        
        #select largest probability class:
        if posteriors[0] >= posteriors[1]: ind = 0
        else: ind = 1
                
            
        
        return outputClasses[ind], posteriors[ind]
        pass
    def test(fileName):
        """
        read in an input test file, and print out 
        """
        pass
    

class TAN():
    """
    implementation of a TAN (tree-augmented naive Bayes) binary classifier
    """
    pass

#---------------------------- utility -----------------------------------------
def importarff(fileName):
    """
    processes an input .arff file containing only discrete variables, into a np.array
    system matrix of the following form:
        
        np.array -   atr1   atr2    atr3   classLabel
        instance1  [ d11    d12    d13 ...
        instance2    d21    d22    d23 ...
        instance3    d31    d32    d33 ...]
    
    input: fileName to be processed
    output: data , attributes(useful for printing)
    
    ex) ['a' ,'b', 'c'] in data -> [0,1,2] in dataMat
    """
    #open data file
    data = arff.load(open(fileName,'rb'))
    
    #make list of lookup dictionaries for each of the attributes - including class label
    atrLU = []  # attribute look up  
    for atr in data['attributes']:
        atrDict = {}
        for n, discreteOption in enumerate(atr[1]):
            atrDict[discreteOption] = n
        atrLU.append(atrDict)
    
    #make a list of lists - [atrID][discreteChoiceID]
    atrDC = []
    for atr in data['attributes']:
        atrDC.append(range(len(atr[1])))
    
    #place transformed data into np.array dataMat
    nInstances  =  len(data['data'])
    nAttributes = len(data['data'][0])
    dataMat = np.zeros([nInstances,nAttributes])
    
    for InstID, instance in enumerate(data['data']):
        for atrID, attribute in enumerate(instance):
            #use dictionary to apply transformation
            dataMat[InstID,atrID] = atrLU[atrID][attribute] 
    
    #make an attribute dictionary for use in training]
    atr = {}
    atr['names'] = data['attributes']
    atr['lookUp'] = atrLU              #lookUp
    atr['DC']     = atrDC              #discreteChoice
    
            
    return dataMat , atr
                    
#---------------------------- plotting ----------------------------------------

#---------------------------- grading -----------------------------------------
#---------------------------- debugging ---------------------------------------
nb = NaiveBayes('vote_train.arff')