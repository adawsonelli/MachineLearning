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
    def __init__(self, fileName, dataArray = None):
        """
        inputs: fileName of training set data file
        """
        #handle instantiation with sub-array of data
        if type(dataArray) == np.ndarray:
            self.data = dataArray
            garbage , self.atr = importarff(fileName)
        
        #handle fileName instantiation
        if dataArray is None:
            self.data , self.atr = importarff(fileName)
        
        #process data  input file:
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
            classInstances = np.zeros([2])  # [negCount ,posCount]
            for instance in self.data: 
                if instance[-1] == 0: #neg
                    classInstances[0] += 1
                elif instance[-1] == 1: #pos
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
     
        #setup output file
        f = open("output.txt",'w')  #write over existing file
        space = " "
        
        #print header
        CLid = len(self.atr['names']) - 1
        for ID, name in enumerate(self.atr['names']):
            if ID != CLid:
                f.write(name[0] + space + "class" + "\n")
        
        #make space
        f.write(" \n")
        
        #make prediction based on training data and print
        correct = 0
        for x in data:
            CL, postProb = self.predictInstance(x)
            
            CLNames = self.atr['names'][-1][1]
            groundTruth = CLNames[int(x[-1])]  ; predicted = CLNames[CL]
            f.write(str(predicted) + space + str(groundTruth) + space +  str(postProb) + "\n")
            
            if groundTruth == predicted: 
                correct += 1
        
        #print number of correct classifications
        f.write("\n")
        f.write(str(correct))
        #close file
        f.close()
        
    def cvTest(self,testData):
        """
        returns accuracy for cross validation
        """
        correct = 0
        for x in testData:
            CL, postProb = self.predictInstance(x)
            if CL == x[-1]: 
                correct += 1
        return float(correct) / testData.shape[1]
            
            
        
    

class TAN():
    """
    implementation of a TAN (tree-augmented naive Bayes) binary classifier
    """
    def __init__(self, fileName, dataArray = None):
        """
        inputs:fileName of training set data or training data array
        """
        #handle instantiation with sub-array of data
        if type(dataArray) == np.ndarray:
            self.data = dataArray
            garbage , self.atr = importarff(fileName)
        
        #handle fileName instantiation
        if dataArray is None:
            self.data , self.atr = importarff(fileName)
            
        self.Pdict = {}
        self.nb = NaiveBayes(fileName,dataArray)  #Naive bayes instance - composition
        self.root = self.primGrowTree()
        
        
    def I(self, xi, xj):
        """
        Conditional Mutual Information - used to assess degree to which two features are 
        dependent
        inputs:
            xi - index of i_th feature
            xj - index of j_th feature
        outputs: 
            CMI - float defining Conditional Mutual Information
        """
        #setup variables for sumation
        CMI = 0 ; P = self.P ; yID = self.nb.nAttributes 
        for y in self.atr['DC'][yID]:
            for ci in self.atr['DC'][xi]:          #choice i
                for cj in self.atr['DC'][xj]:      #choice j
                    CMI += P([(xi,ci),(xj,cj),(yID,y)]) *  \
                   np.log2(P([(xi,ci),(xj,cj)],[(yID,y)])/(P([(xi,ci)],[(yID,y)])*P([(xj,cj)],[(yID,y)])))
        return CMI
    
    
    def P(self,X,Y = []):
        """
        determine the conditional probability P(X|Y) from the training dataset
        with laplacian corrections
        inputs:
            X: list of feature ID - choice tuples connected by & relationship
            Y: list of feature ID - choice tuples that the probability is conditioned on
        example:  
            x1 is feature ID , c1 is choice ID
            P([(x1,c0),(x7,c1)] ,[(YID,c0),(x4,c4)])
        """
        #check if P has been calculated before, and if so, return it
        name = self.Pname(X,Y)
        if name in self.Pdict:
            return self.Pdict[name]
        
        #calculate P and add it to dictionary
        #generate counts for num and den of probability
        num = 0 ; den = 0 
        for instance in self.data:
            
            #handle numerator increment - only increment if all conditions met
            ni = 1
            for feature in X: ni = ni and (instance[feature[0]] == feature[1])  
            for feature in Y: ni = ni and (instance[feature[0]] == feature[1])
            if ni: num += 1
            
            #handle denominator increment only if conditional contributions met
            di = 1
            for feature in Y: di = di and (instance[feature[0]] == feature[1])
            if di: den += 1
        
        #handle laplacian correction
        xChoiceProducts = 1
        for feature in X: xChoiceProducts = xChoiceProducts * len(self.atr['DC'][feature[0]])
        P = float(num + 1) / (den + xChoiceProducts)
        
        #add P to Pdict - store smart state to avoid recalculation
        self.Pdict[name] = P
        
        return P
    
    def Pname(self,X,Y):
        """
        determines and returns a string name that is unique to a particular 
        conditional probabily, defined by X and Y
        """
        name = ""
        for feature in X: name = name + str(feature[0]) + str(feature[1])
        name = name + "|"
        for feature in Y: name = name + str(feature[0]) + str(feature[1])
        
        return name 
        
        
        
        
    
    def primGrowTree(self):
        """
        grow a Maximal Spanning Tree (MST) using prims algorithm, and CMI as
        a metric for connection strength
        """
        #utility fxns
        def popNode(nodeList,nID): 
            """returns and removes node from list""" 
            for ID, node in enumerate(nodeList):
                if node.ID == nID:
                    return nodeList.pop(ID)
        def getNode(nodeList,nID): 
            """returns node without removing from list"""
            for ID, node in enumerate(nodeList):
                if node.ID == nID:
                    return node
        
        def sortNodeList(nodeList):
            """
            perfroms in place selection sort of  a nodeList
            """
            for sortedID, node in enumerate(nodeList):
                minVal = nodeList[sortedID].ID
                minID = sortedID
                for entryID in range(sortedID,len(nodeList)):
                    if nodeList[entryID].ID < minVal:
                        minVal = nodeList[entryID].ID
                        minID = entryID
                
                #swap items:
                nodeList[sortedID], nodeList[minID] = nodeList[minID], nodeList[sortedID]
            #return nodeList
  
        #form node for each feature, not including Class Label
        free = [] ; network = []
        for fID in range(self.nb.nAttributes):
            free.append(node(fID))
        
        #establish first feature as root
        root = popNode(free, 0)
        network.append(root)
        
        
        #grow tree:
        growTree = True
        while growTree:
            #break condition:
            if len(network) == self.nb.nAttributes: growTree = False ; break
            
            CMImx = 0 ; nNodeID = -1; fNodeID = -1
            for nNode in network:
                for fNode in free:
                    if self.I(nNode.ID,fNode.ID) > CMImx:  #this should handle tie breaking criteria natively!
                        CMImx = self.I(nNode.ID,fNode.ID)
                        nNodeID = nNode.ID 
                        fNodeID = fNode.ID
            #add best node to network
            fBest = popNode(free,fNodeID)
            nBest = getNode(network,nNodeID)
            nBest.next.append(fBest)
            network.append(fBest)
            sortNodeList(network)                         #ensure correct order of network for tiebreaks
            
        return root
    
    def mostImportantDependency(self,fID):
        """
        Determine the most important dependancy of the feature specified by fID
        using the tree located at self.root
        inputs:
            fID -[int] ID of the feature who's MID you are trying to find
        outputs:
            pID -[int] ID of the parent of the requested feature. if root, return
            None
        """
        if fID == 0: #handle root:
            return None
        return self.root.returnParent(fID)
            

            
    def predictInstance(self,x):
        """
        use trained TAN classifier to predict the class of an input
        instance
        input: 1d np.array vector
        outputs: predictedClass, probability
        """
        #trim off class label and prep
        x = x[0:-1] ; posteriors = [] ; outputClasses = [0,1] ; yID = self.nb.nAttributes
        
        # calculate posterior probabilities
        for hyp in outputClasses: 
            P = 1.0
            #p(x | h) = p(d1 | h) * p(d_2 | h ) * ... p(d_n | h )
            for atrID,choice in enumerate(x):
                #form conditional probability expressions
                X = [(atrID,choice)]
                Y = [(yID,hyp)]
                #add most important condition to Y
                depID = self.mostImportantDependency(atrID)
                if depID != None:
                    Y.append((depID,x[depID]))
                
                #calculate
                P = P*self.P(X,Y)
            
            #solve for posterior
            P = (P*self.nb.classPriors[hyp])
            
            #add to list
            posteriors.append(P)
        
        
        # normalize poseterior probabilities
        posteriors = [posteriors[0] / sum(posteriors) , posteriors[1] / sum(posteriors)]
        
        #select largest probability class
        if posteriors[0] >= posteriors[1]: ind = 0
        else: ind = 1
        
        return outputClasses[ind], posteriors[ind]
    
    
    def test(self, fileName):
        """
        read in an input test file, and print out 
        """
        #process input file
        data, atr = importarff(fileName)
        
        #setup output file
        f = open("output.txt",'w')  #write over existing file
        space = " "
        
        #print header
        CLid = len(self.atr['names']) - 1
        for ID, name in enumerate(self.atr['names']):
            if ID == 0: #root
                f.write(name[0] + space + "class" + "\n")
                
            elif ID != CLid:
                parentID =  self.mostImportantDependency(ID)
                parentName = self.atr['names'][parentID][0]
                f.write(name[0] + space + parentName + space + "class" + "\n")
        
        #make space
        f.write(" \n")
        
        #make prediction based on training data:
        correct = 0
        for x in data:
            CL, postProb = self.predictInstance(x)
            
            CLNames = self.atr['names'][-1][1]
            groundTruth = CLNames[int(x[-1])]  ; predicted = CLNames[CL]
            f.write(str(predicted) + space + str(groundTruth) + space +  str(postProb) + "\n")
            
            if groundTruth == predicted: 
                correct += 1
                
        #print number of correct classifications
        f.write("\n")
        f.write(str(correct))
        
        #close file
        f.close()
    
    def cvTest(self,testData):
        """
        returns accuracy for cross validation
        """
        correct = 0
        for x in testData:
            CL, postProb = self.predictInstance(x)
            if CL == x[-1]: 
                correct += 1
        return float(correct) / testData.shape[1]
            
              
        
class node():
    """
    prim algorithm node for forming a-cylic graph
    """
    def __init__(self,ID):
        self.ID = ID         #ID number, aka feature number
        self.next = []
        self.inNetwork = False
    
    def cprint(self,ntabs = 0):
        """
        print tree struct to console for viewing
        """
        tab = "|    "
        print(ntabs*tab  +  str(self.ID) + '\n'),
        if self.next == [] : return 
        
        for node in self.next:
            node.cprint(ntabs + 1)
    
    def returnParent(self,childID):
        """
        search dependency tree and return parent of requested child 
        """
        if self.next == []:
            return 0
        
        parent = False
        for child in self.next:
            if child.ID == childID:
                parent == True
                return self.ID
       
        
        if not parent:
            for child in self.next:
                ID = child.returnParent(childID)
                if  ID != 0:
                    return ID
            return 0
        

        
        
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
def crossValidationPlot(fileName):
    """
    perform a 10-fold cross validation on the specified file with both naive bayes
    and TAN algorithms, and output model accuracy 
    """
    #process input file
    data, atr = importarff(fileName)
    Folds = []        #init empty fold set
    foldIDs = []
   
    #make lists of positive and negative instance ID's
    posIDs = [] ; negIDs = []
    for ID,instance in enumerate(data):
        if instance[-1] == 0:   #negative
           negIDs.append(ID)
        elif instance[-1] == 1: #pos
            posIDs.append(ID)
    
    #state information
    nInstances = len(data)
    nFolds = 10
    #make n fold groups, each are np.arrays
    instancesPerGroup = nInstances / nFolds  
    
    #separate instances into groups
    for grp in range(nFolds):
        #determine sampling numbers for this group
        posPercent = len(posIDs)/float(nInstances)
        nInstances = intSelect(instancesPerGroup)
        nPos       = intSelect(nInstances * posPercent)
        nNeg       = intSelect(nInstances * (1-posPercent))
        
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
        group = np.zeros([len(groupSamples),data.shape[1]])
        for newID,sampleID in enumerate(groupSamples):
            group[newID,:] = data[sampleID,:]
            
            
        #add group to folds, instance ID's to fold ID's 
        Folds.append(group)
        foldIDs.append(groupSamples)
    
    #perform the cross validation:
    results = np.zeros([nFolds,2])
    for fID, fold in enumerate(Folds):
        folds = Folds[:] #make a shallow copy
        testSet = folds.pop(fID)
        trainingSet = merge(folds)
        #train learners
        nb = NaiveBayes(fileName,trainingSet)
        tan = TAN(fileName,trainingSet)
        #store results
        results[fID,0] =  nb.cvTest(testSet)
        results[fID,1] = tan.cvTest(testSet)
        print('Fold ' + str(fID))
    
    return results
        

def flatten(nestedList):
    """
    method for flattening nested lists
    """
    flattenedList = []
    for sublist in nestedList:
        for val in sublist:
            flattenedList.append(val)
    return flattenedList

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
    allFolds = np.array([]).reshape(0,foldList[0].shape[1])  #get into the right shape!!
    for fold in foldList:
        allFolds = np.vstack((allFolds,fold))
    return allFolds

      
      
#---------------------------- grading -----------------------------------------
def grading(trainFile,testFile, learningMethod):
    """
    called from the command line for grading purposes
    """
    if learningMethod == "n":
        nb = NaiveBayes(trainFile)
        nb.test(testFile)
    elif learningMethod == "t":
        tan = TAN(trainFile)
        tan.test(trainFile)
        
    
#---------------------------- debugging ---------------------------------------
#nb = NaiveBayes('vote_train.arff')
#nb = NaiveBayes('lymph_train.arff')

#tan = TAN('vote_train.arff')
tan = TAN('lymph_train.arff')  
#tan.P([(13,6)],[(0,2),(18,0)]) 



