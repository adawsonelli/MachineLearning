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
        
        #train probabilities:                                                #http://www.saedsayad.com/naive_bayesian.htm
        self.classInstances , self.classPriors = self.train("classPriors")   #Class Prior Probability and instances
        self.likelyhoods = self.train("likelyhoods")                         #conditional probabilities
        self.predictorPrior = self.train("predictorPrior")                   #predictor prior probabilities
    
    def train(self,Tset):
        """
        train a naive bayes classifier - both class probabilites and atrProbs
        structure: 
            self.classPriors     = np.array[1,2]
            self.classInstances  = np.array[1,2]
            self.likelyhoods = [nHyp][nAttributes][nChoices] (nChoices will vary by atr)
			  self.predictorPrior =   [nAttributes][nChoices]
        """
        # train class prior probabilities: P(O = T) and P(O = F)
        if Tset == "classPriors":  #assumes binary classification problem
            
            #count class instances:
            classInstances = np.zeros([2])  # [posCount ,negCount]
            for instance in self.data: 
                if instance[-1] == 0: #pos
                    classInstances[0] += 1
                elif instance[-1] == 1: #neg
                    classInstances[1] += 1
            
            # calculate class probabilities (using laplace correction)
            classPriors = np.zeros([2])
            for ID, CI in enumerate(classInstances):
                classPriors[ID] = float(classInstances[ID] + 1) / (self.nInstances + self.nClasses)
                
            return classInstances, classPriors
        
        #train likelyhoods P(A_n | Y)
        elif Tset == "likelyhoods":
            
            likelyhoods = []  # [1x2]
            for outputClass in range(self.nClasses):  #binary classification for now
                #calculate counts(A_n and Y):
                P_AnY = []
                for atrID, attribute in enumerate(self.atr['DC']):
                    nChoices = len(attribute)
                    atrProbs = []
                    for choice in attribute:
                        count = 0
                        #loop over data and count instances:
                        for instance in self.data:
                            if (instance[atrID] == choice) and instance[-1] == outputClass:
                                count += 1
                        #calculate attribute probability:
                        P = float(count + 1) / (self.classInstances[outputClass] + nChoices )
                        #assemble lists:
                        atrProbs.append(P)
                    P_AnY.append(atrProbs)
                likelyhoods.append(P_AnY)
            return likelyhoods 
        
        #train predictor prior probabilities
        elif Tset == "predictorPrior":
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
    
    def priorCheck(self):
        """
        check that that weighted sum of the likelyhoods leads to the predictor prior
        """
        priorCheck = []
        #calculate with weighted sum of likelyhoods
        atrProbs = []
        for atrID, atr in enumerate(self.predictorPrior):
            choiceProbs = []
            for choiceID, choice in enumerate(atr):
                p = self.likelyhoods[0][atrID][choiceID] * self.classPriors[0] +  \
                    self.likelyhoods[1][atrID][choiceID] * self.classPriors[1]
                choiceProbs.append(p)
            atrProbs.append(choiceProbs)
        return atrProbs
                    
        
            
        
    def predictInstance(self,x):
        """
        use trained naive Bayes binary classifier to predict the class of an input
        instance
        input: 1d np.array vector
        outputs: predictedClass, probability
        """
        #trim off class label:
        x = x[0:-1]
        
        #solve for the Predictor Prior Probability for x:
        Px = 1
        for atrID,choice in enumerate(x):
            Px = Px * self.predictorPrior[atrID][int(choice)]
            
        #generate posterior probabilities for each class:
        posteriors = []
        outputClasses = [0,1]
        
        for hyp in outputClasses: #select h
            P = 1.0
            #p(x | h) = p(d1 | h) * p(d_2 | h ) * ... p(d_n | h )
            for atrID,choice in enumerate(x):
                P = P*self.likelyhoods[hyp][atrID][int(choice)]
            #solve for posterior
            P = (P*self.classPriors[hyp]) / Px
            
            #add to list
            posteriors.append(P)
        
        
        # normalize to sum of 
        posteriors = [posteriors[0] / sum(posteriors) , posteriors[1] / sum(posteriors)]
        
        #select largest probability class:
        if posteriors[0] >= posteriors[1]: ind = 0
        else: ind = 1
        
        return outputClasses[ind], posteriors[ind]
    
    
    def test(self, fileName):
        """
        read in an input test file, and print out 
        """
        #process input file
        data, atr = importarff(fileName)
        
        #print header
        
        #setup output file
        f = open("output.txt",'w')  #write over existing file
        #f.write("output of nFold Stratified Cross-Validation \n")
        #f.write(" \n")
        #f.write("fold predicted ground truth confidence \n")
        space = "    "
        
        #make prediction based on training data:
        for x in data:
            CL, postProb = self.predictInstance(x)
            
            CLNames = self.atr['names'][-1][1]
            groundTruth = CLNames[int(x[-1])]  ; predicted = CLNames[CL]
            f.write(str(predicted) + space + str(groundTruth) + space +  str(postProb) + "\n")
        
        #close file
        f.close()
            
            
        
    

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
nb.predictInstance(data[0,:])