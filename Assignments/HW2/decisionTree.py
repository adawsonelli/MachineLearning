"""
ID3
"""
#---------------------- system design documentation ---------------------------
# -one data object is created to represent all training instances in the system
#  each node has a reference to this data object via self.data
# -which instances and which attributes belong in each node are represented by the
#  nodes internal self.instanceIDs and self.atrIDs.
#
# -This design has the advantage of not creating copies of the system at each node, 
#  but the disadvantage of not being able to make use of np vectorized code
# 
# -another option that was not explored but could have been was using a more efficient
#  data representation by replacing all nominal values with integers

#data structures:
#                            F_1  F_2  F_3 ... F_n
#   data =   [  instance_1[                         ]
#               instance_2[                         ]
#               instance_n[                         ]      ]  
#
# implmented as lists of lists - data type is either float or unicode

#-------------------------- imports -------------------------------------------
import arff
import numpy as np
from matplotlib import pyplot as plt



#------------------------ load data -------------------------------------------
#data = arff.load(open('weather.arff','rb'))
#data = arff.load(open('heart_train.arff','rb'))


#-------------------------- Node class ----------------------------------------

class Node:
    """
    node forms the basic unit of the a tree 
    """     
    def __init__(self, data , m , parent = None, atrIDs = None , instanceIDs = None):
        #when initializing first top level node
        if atrIDs == None and instanceIDs == None: 
            atrIDs = range(len(data['attributes']) - 1)
            instanceIDs = range(len(data['data']))
         
        self.parent = parent                           #parent node
        self.data = data                               #reference to system data dictionary
        self.m = m                                     #stopping criteria
        self.children = []                             #children nodes
        self.isLeaf = False
        self.classLabel = None                         #if leaf - define a class label
        self.atrIDs = atrIDs                           #attributes this node is awair of
        self.instanceIDs = instanceIDs                 #list of training instance ID's this node contains
        self.split = None                              #which attribute does this node split on?
        self.nInstances = len(self.instanceIDs)
        self.classCol = len(self.data['data'][0]) - 1  #last column index is data
        self.pos = self.count('pos')             
        self.neg = self.count('negative')
    
    def count(self,label):
        """
        deterine the number of negative or positive instances in the set
        """
        if label == 'pos': classValueID = 0
        if label == 'negative': classValueID = 1
        
        count = 0
        for i in self.instanceIDs:
            if self.data['data'][i][self.classCol] == self.data['attributes'][self.classCol][1][classValueID]:
                count += 1
        return count
        
    def Hd(self,positive,instances):  #assumes binary class label
        if positive == 0 or instances == 0:  #divde by zero error
            return 0
        Py = float(positive) / instances ;  Pyc = 1 - Py
        if Py == 0 or Pyc == 0:  #handle the case when log would blow up
            return 0
        H = - (Py * np.log2(Py) + Pyc * np.log2(Pyc))
        return H
        
    def entropy(self):
        """
        calculates entropy of class label outcomes at this node
        """
        return self.Hd(self.pos,self.nInstances)
       
        
    def conditionalEntropy(self,atrID,threashold):
        """
        calculates conditional entropy of attributes given the conditions
        inputs:
             atrID - integer ID of the atribute to be evaluated
             threashold - if atr is nominal, threashold is discarded
                          if atr is numeric, threashold is proposed threashold
        """
        #nominal / discrete type attributes:
        if type(self.data['attributes'][atrID][1]) == list: 
            choices = self.data['attributes'][atrID][1]
           
            #calculate training instances per choice and positive results per choice
            counts = [] ; positives = []
            for choice in choices:  
                count = 0 ; positive = 0
                for ID in self.instanceIDs:
                    if self.data['data'][ID][atrID] == choice:
                        count += 1
                        if self.data['data'][ID][self.classCol] == self.data['attributes'][self.classCol][1][0]:
                            positive += 1
                counts.append(count)
                positives.append(positive)
            
            #calculate weighted entropy per choice (I.E. conditional entropy on atrID)  [3.27]
            wHd = 0
            for i, choice in enumerate(choices):
                wHd += (float(counts[i])/self.nInstances) * self.Hd(positives[i],counts[i])
      
        
        #numeric type attributes
        if type(self.data['attributes'][atrID][1]) == unicode:
            #calculate training instances on either side of the threashold and positives on either side
            counts = [0,0]; positives = [0,0]  #[<=, >] 
            for ID in self.instanceIDs:
                if self.data['data'][ID][atrID] <= threashold:
                    counts[0] += 1
                    if self.data['data'][ID][self.classCol] == self.data['attributes'][self.classCol][1][0]:
                         positives[0] += 1
                else:
                    counts[1] += 1 
                    if self.data['data'][ID][self.classCol] == self.data['attributes'][self.classCol][1][0]:
                         positives[1] += 1
            #calculate weighted entropy for numerical attribute on this threashold:
            wHd = 0
            for i , condition in enumerate(['belowThreashold','aboveThreashold']):
                wHd += (float(counts[i])/self.nInstances) * self.Hd(positives[i],counts[i])
            
        return wHd
               
                    
            
    def infoGain(self,atrID, threashold = False):
        """
        calculates the info gain of an attribute (specified by attribute ID) with
        respect to the node's instances
        """
        return self.entropy() - self.conditionalEntropy(atrID,threashold)
        
    
    def determineCandidateSplits(self):
        """
        determine from an internal list of attributes, which are candidates to split on.
            -each nominal attribute (which has not been used before) is one candidate
            -each numeric attribute is n candidates, where n is the number of class transitions
            in an ordered list of instances D on numeric attribute Xi
            
        returns:
            a list of candidate attribute ID's (nominal) and candidate attribute ID's + threashold
            ex) [1 , [2 ,22.2] , [2,34.0], 4 , 5] 1,4,5 are nominal features, 2 is numerical
        """
        candidateSplits = [] 
        for candidateSplit in self.atrIDs:
            #nominal
            if type(self.data['attributes'][candidateSplit][1]) == list:
                candidateSplits.append(candidateSplit)  #all nominal types added once
                
            #numeric 3.12
            if type(self.data['attributes'][candidateSplit][1]) == unicode:
                #form a matrix Xi [numericFeature_col | class_col]
                Xi = np.zeros([self.nInstances,2])
                #create a lookup table for class labels (binary)
                classDict = {self.data['attributes'][self.classCol][1][0]:1 ,  #negative
                             self.data['attributes'][self.classCol][1][1]:0}   #positive
                #populate matrix
                for newID, instanceID in enumerate(self.instanceIDs):
                    Xi[newID,0] = self.data['data'][instanceID][candidateSplit]
                    Xi[newID,1] = classDict[self.data['data'][instanceID][self.classCol]]
                #perform argsort, get indecies that would sort the array
                sortIDs = np.argsort(Xi[:,0])
                #form groups of same value numerical features and class change indicators:
                #   0 == negative, 1 == positive, 2 = both in same group
                groups = []
                currentGroupLabel = Xi[sortIDs[0],0]                #value of attribute
                currentGroup = [Xi[sortIDs[0],0],Xi[sortIDs[0],1]]  #[value, class change indicator]
                for ID in sortIDs:
                    if Xi[ID,0] != currentGroupLabel:
                        groups.append(currentGroup)
                        currentGroupLabel = Xi[ID,0]
                        currentGroup = [Xi[ID,0],Xi[ID,1]]
                    if Xi[ID,0] == currentGroupLabel:
                        if currentGroup[1] != Xi[ID,1]:  #2 member of group in diff classes
                            currentGroup[1] = 2
                #add on last group if it was missed (may allways be missed?)
                if groups[-1:] != currentGroup:  #end
                    groups.append(currentGroup)
                    
                #form threasholds at the midpoint of disimilar groups
                threasholds = []
                for i in range(len(groups) - 1):
                    if groups[i][1] != groups[i+1][1] or (groups[i][1] == 2 or groups[i+1][1] == 2): #class boundry
                        threasholds.append((groups[i][0] + groups[i+1][0])/2)
                
                #append [candidate split , threashold] to list of candidate splits
                for threashold in threasholds:
                    candidateSplits.append([candidateSplit , threashold])
        return candidateSplits 
                    

    def stoppingCriteria(self,candidateSplits = False):
        """
        stopping criteria for making a leaf node are:
            (i)   all training instances belonging to the node are the same class
            (ii)  there are fewer than m training instances reaching the node
            (iii) no feature has positive info gain
            (iv)  there are no more remaining candidate splits
        
        return: True if should form a leaf, return false otherwise.
        """
        #first check for stopping criteria - doen't require candidateSplit info
        if candidateSplits == False:
            #(i) all training instances belonging to the node are the same class
            if self.pos == 0 or self.neg == 0:
                return True
            
             #(ii)  there are fewer than m training instances reaching the node
            if self.nInstances < self.m:
                return True
        
        #second check for stopping criteria - requires candidate split info
        else:             
            #(iii) no feature has positive info gain
            infoGainSum = 0
            for split in candidateSplits:
                #nominal feature
                if type(split) == int:
                    infoGainSum += self.infoGain(split)
                #numeric feature
                if type(split) == list:
                    infoGainSum += self.infoGain(split[0],split[1])
            
            if infoGainSum == 0:
                return True
                    
            #(iv)  there are no more remaining candidate splits
            if len(self.atrIDs) == 0 or len(candidateSplits) == 0:
                return True
            
            #none of the stopping criteria have been met
        return False
    
    def findClassLabel(self):
        """
        All nodes calling this function are leaf nodes.The class label of a 
        node is determined as follows:
            
            (i)   in the case where there are an unequal number of instance 
                  of the two classes, predict the most common class
            (ii)  in the case where there are an equal number of instances
                  of the two classes, predict the most common class of the parent
            (iii) if the number of instances in the node is zero, the leaf should
                  predict the most common class of the parent
        
        returns: class label 
        """
        #(i) 
        if self.pos > self.neg:
            return self.data['attributes'][self.classCol][1][0]
        elif self.pos < self.neg:
            return self.data['attributes'][self.classCol][1][1]
        
        #(ii) & (iii) 
        elif self.pos == self.neg: 
            return self.parent.findClassLabel()
            

    def findBestSplit(self, candidateSplits):
        """
        score each of the candidate splits in terms of info gain and return best split
        returns:
            if nominal:atrID of highest info gain 
            if numeric:[atrID, threashold]
        """
        bestSplit  = {} ; bestSplit['infoGain'] = -1000 ; bestSplit['IDs'] = []
        for cs in candidateSplits:
            #nominal
            if type(cs) == int:
                infoGain = self.infoGain(cs)
            #numeric
            elif type(cs) == list:
                infoGain = self.infoGain(cs[0],cs[1])
            
            #handle infogain
            if infoGain == bestSplit['infoGain']: # we have a tie!
                bestSplit['IDs'].append(cs)
            if infoGain > bestSplit['infoGain']:
                bestSplit['IDs'] = []
                bestSplit['IDs'].append(cs)
                bestSplit['infoGain'] = infoGain
            
        #break ties by always taking the first added to the list
        return bestSplit['IDs'][0]
    
    
    def makeSubtree(self):   #[3.9]
        """
        starting from current node, make a subtree recursively
        """
        #first stopping criteria check - doesn't require candidate splits
        if self.stoppingCriteria():
             self.isLeaf = True
             self.classLabel = self.findClassLabel()
             return
             
        #calculate candidate splits
        candidateSplits = self.determineCandidateSplits()
        
        #second stopping criteria check - requires candidate splits
        if self.stoppingCriteria(candidateSplits):
            self.isLeaf = True
            self.classLabel = self.findClassLabel()
        else:
            split = self.findBestSplit(candidateSplits) #feature ID
            self.split = split
            
            #split on nominal feature
            if type(split) == int:
                splitID = split
                nChoices = len(self.data['attributes'][splitID][1])
                sortedInstances = []
                for child in range(nChoices):sortedInstances.append([])
                for ID in self.instanceIDs:
                    for i, attribute in enumerate(self.data['attributes'][splitID][1]):
                        if self.data['data'][ID][splitID] == attribute:
                            sortedInstances[i].append(ID)
                            
            #split on numeric feature - guarenteed bifercation
            if type(split) == list:
                splitID = split[0] ; threashold = split[1]
                sortedInstances = [[],[]]    #[[lower],[upper]]
                for ID in self.instanceIDs:
                    if self.data['data'][ID][splitID]   <= threashold:
                        sortedInstances[0].append(ID)
                    elif self.data['data'][ID][splitID]  > threashold:
                        sortedInstances[1].append(ID)
            
            
            #for each outcome k in S - make a new node and add to children
            childAtrIDs = self.atrIDs[:] #can't split on this feature again (is this true for numerics?!)
            #childAtrIDs.remove(splitID)
            for childInstanceIDs in sortedInstances:
                childNode = Node(self.data,self.m,self,childAtrIDs,childInstanceIDs)
                self.children.append(childNode)
            
            #for each child, make a subtree
            for child in self.children:
                child.makeSubtree() 
        
        return
            
    
    def printTree(self,f = "outputTree.txt", nTabs = 0):
        """
        prints the contents of the trained tree into a file
        inputs:
            f - either fileName or a file object
            tabs - int number of tabs
        """
        #handle input file
        if type(f) == file:
            pass
        elif type(f) == str:
            f = open("outputTree.txt",'w')
            
        #handle basecase - leaf node
        if self.isLeaf:
            return
       
        #build string and write to file
        for childID, child in enumerate(self.children):
            #build printing string
            strTab = nTabs * '|    '
            
            #nominal split
            if type(self.split) == int:
                splitName =         self.data['attributes'][self.split][0]
                childChoice = ' = ' + self.data['attributes'][self.split][1][childID]
            #numeric split -> bifercation
            elif type(self.split) == list:
                splitName = self.data['attributes'][self.split[0]][0]
                if childID == 0: #lower
                    childChoice =  ' <= '  + str(self.split[1])
                if childID == 1: #upper
                    childChoice =  ' >  '  + str(self.split[1])
                    
                
            #determine classification split at this node
            classSplit = "[" + str(child.pos) + ' ' + str(child.neg) + "]"
            
            #leaf node -> report class prediction
            if child.isLeaf:
                leaf = ":" + str(child.classLabel)
                f.write(strTab + splitName + childChoice + classSplit + leaf + "\n")
                child.printTree(f,nTabs + 1)
            else:
                f.write(strTab + splitName + childChoice + classSplit + "\n")
                child.printTree(f,nTabs + 1)
        
        
    
    def classify(self,instance):
        """
        take an unknown instance make a prediction about it's class
        inputs:
            instance - a list of attributes
        outputs:
            a prediction of class
        """
        if self.isLeaf:
            return self.classLabel
        
        #nominal feature
        if type(self.split) == int:
            featureValue = instance[self.split]
            for i, child in enumerate(self.children):  #children are stored in the order the attributes appear
                if featureValue == self.data['attributes'][self.split][1][i]:
                    return child.classify(instance)  #enter that child node
        
        #numeric feature
        if type(self.split) == list:
            featureValue = instance[self.split[0]]
            if featureValue <= self.split[1]:
                return self.children[0].classify(instance)
            if featureValue > self.split[1]:
                return self.children[1].classify(instance)
        
        

#-------------------------- test set functions --------------------------------
def evaluateAccuracy(root, Data):
    """
    takes in a dataset and returns the percentage of instances properly classified
    """
    correct = 0
    for i, instance in enumerate(Data['data']):
        classification = root.classify(instance)
        if str(instance[root.classCol]) == str(classification):
            correct += 1
    return float(correct) / len(Data['data'])


def runTestSet(fileName = 'heart', m = 2):
    """
    trains tree, then loads and runs test set given filename
    """
    
    #train tree
    training = fileName + '_train.arff'
    trainingData = arff.load(open(training,'rb'))
    
    root = Node(trainingData, m)
    root.makeSubtree()
    root.printTree()
    
    #run test set
    testing = fileName + '_test.arff'
    testingData = arff.load(open(testing,'rb'))
    f = open("outputTree.txt",'a')  #append to existing file
    f.write("<Predictions for the Test Set Instances> \n")
    correct = 0
    for i, instance in enumerate(testingData['data']):
        classification = root.classify(instance)
        f.write(str(i + 1) + ": Actual: " + str(instance[root.classCol]) + " Predicted: " + str(classification) + "\n" )
        if str(instance[root.classCol]) == str(classification):
            correct += 1
    #print total sucess rate
    f.write("Number of correctly classifed: " + str(correct) + " Total number of test instances: " + str(len(testingData['data'])))
    

def outputTestResults(trainingFileName, testFileName, m):
    """
    trains tree, then loads and runs test set given filename
    """
    #train tree
    trainingData = arff.load(open(trainingFileName,'rb'))
    
    root = Node(trainingData, m)
    root.makeSubtree()
    root.printTree()
    
    #run test set
    testingData = arff.load(open(testFileName,'rb'))
    f = open("outputTree.txt",'a')  #append to existing file
    f.write("<Predictions for the Test Set Instances> \n")
    correct = 0
    for i, instance in enumerate(testingData['data']):
        classification = root.classify(instance)
        f.write(str(i + 1) + ": Actual: " + str(instance[root.classCol]) + " Predicted: " + str(classification) + "\n" )
        if str(instance[root.classCol]) == str(classification):
            correct += 1
    #print total sucess rate
    f.write("Number of correctly classifed: " + str(correct) + " Total number of test instances: " + str(len(testingData['data'])))



    

#-------------------------- generate plots ------------------------------------
def plotLearningCurves(fileName = 'heart', m = 4):  #part 2
    """
    plot the learning curve which shows predictive accuracy vs training set size
    for a constant m = 4
    """
    
    #setup dataset over which numerical experiment will take place
    training = fileName + '_train.arff'
    trainingData = arff.load(open(training,'rb'))
    
    #setup testing set
    testing = fileName + '_test.arff'
    testingData = arff.load(open(testing,'rb'))
    
    
    
    #loop over specified percentages
    percentages = [.05,.1,.2,.5,1.0]
    results = [[],[],[],[],[]]
    trainingSetLen = len(trainingData['data'])
    for i , p in enumerate(percentages):
        
        #random samples at specified percentage
        for sample in range(10): #perform 10 times per percentage
                      
            #make a local copy of training data
            Data = trainingData.copy()
            Data['data'] = []             #clean data set
            
            #randomly select n*p samples from training set, without replacement - unique samples
            sampleIndecies = range(trainingSetLen)
            nSamples = int(np.floor(p*trainingSetLen))
            for s in range(nSamples):
                setLen = len(sampleIndecies)
                popID = np.random.randint(0,setLen)
                
                #draw one datapoint without replacement
                sampleID = sampleIndecies.pop(popID)
                
                #add datapoint to training data
                Data['data'].append(trainingData['data'][sampleID])
        
            #train tree
            root = Node(Data, m)
            root.makeSubtree()
            
            #test tree for accuracy
            accuracy = evaluateAccuracy(root,testingData)
            
            #store accuracy
            results[i].append(accuracy)
    
    #make plots
    Min = [] ; Max = [] ; mean = []
    for p,percentage in enumerate(percentages):
        Min.append(min(results[p]))
        Max.append(max(results[p]))
        mean.append(sum(results[p])/len(results[p]))
    
    plt.plot(percentages,Min)
    plt.plot(percentages,mean)
    plt.plot(percentages,Max)
    plt.ylabel('accuracy')
    plt.xlabel('percentage of training set')
    plt.title('prediction accuracy vs. percentage of training set @ m = 4')
    plt.grid()
    
    
                
             
    
    
def plotAccuracyVsTreeSize(fileName = 'heart'):
    """
    plot the Accuracy of the tree vs size for the entire training set
    """
    #setup dataset over which numerical experiment will take place
    training = fileName + '_train.arff'
    trainingData = arff.load(open(training,'rb'))
    
    #setup testing set
    testing = fileName + '_test.arff'
    testingData = arff.load(open(testing,'rb'))
    
    
    M = [2,5,10,20]
    accuracy = []
    for m in M:
        root = Node(trainingData, m)
        root.makeSubtree()
        accuracy.append(evaluateAccuracy(root,testingData))
    
    
    #plot m vs. tree accuracy
    plt.plot(M,accuracy)
    plt.plot(M,accuracy,'ro')
    plt.ylabel('accuracy')
    plt.xlabel('m stopping criteria')
    plt.title('prediction accuracy vs. m stopping criteria')
    plt.grid()
    
    
    
        
#---------------------------- tests -------------------------------------------
tests = False
import unittest
if tests:
    data = arff.load(open('weather.arff','rb'))
    root = Node(data,2)



class TestID3(unittest.TestCase):
    def testNode(self):
        
        #test that all internal state information was created properly
        self.assertEqual(root.atrIDs,range(4))
        self.assertEqual(root.instanceIDs,range(14))
        self.assertEqual(root.classCol,4)
        self.assertEqual(root.pos + root.neg,root.nInstances)
    
    def testDetermineCandidateSplits(self):
        cs = [0, [1, 64.5], [1, 66.5], [1, 70.5], [1, 71.5], [1, 73.5], [1, 77.5], [1, 80.5],
             [2, 67.5], [2, 72.5], [2, 82.5], [2, 85.5], [2, 88.0], [2, 90.5], 3]
        self.assertEqual(root.determineCandidateSplits(),cs)
    
    def testEntropy(self):
        self.assertEqual(root.entropy(),0.94028595867063114)
        
#    def testConditionalEntropy(self):
#        self.assertEqual(root.conditionalEntropy(0,False),)
    



if tests:
    unittest.main()
    
    

##------------------------------- main -----------------------------------------   
#root = Node(data,2)
#root.makeSubtree()
#root.printTree()
#results = plotLearningCurves()  
#runTestSet()
    