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
        if dataArray != None:
            self.data = dataArray
            garbage , self.atr = importarff(fileName)
        
        #handle fileName instantiation
        if dataArray == None:
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
        inputs:fileName of training set data
        """
        #handle instantiation with sub-array of data
        if dataArray != None:
            self.data = dataArray
            garbage , self.atr = importarff(fileName)
        
        #handle fileName instantiation
        if dataArray == None:
            self.data , self.atr = importarff(fileName)
        
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
                    CMI += P([(xi,ci),(xj,cj), (yID,y)]) *  \
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
        xsumChoices = 0 ; ysumChoices = 0 
        for feature in X: xsumChoices += len(self.atr['DC'][feature[0]])
        #P = float(num + len(X)) / (den + sumChoices)
        #P = float(num + len(X)) / (den + xsumChoices)
        P = float(num + 1) / (den + xsumChoices)
        #P = float(num + 1) / (den + len(self.atr['DC'][X[0][0]]))  ## correct laplacian correction to agree with outputs
        
        return P
    
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



