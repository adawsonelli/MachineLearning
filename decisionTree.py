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

#-------------------------- imports -------------------------------------------
import arff
import numpy as np
import math


#------------------------ load data -------------------------------------------
data = arff.load(open('weather.arff','rb'))




#-------------------------- classes -------------------------------------------

class node:
    """
    node forms the basic unit of the a tree 
    """
    def __init__(self, data , atrIDs , instanceIDs ):
        self.data                                  #reference to system data dictionary
        self.children = []                         #children nodes
        self.isLeaf = False
        self.classLabel = None                     #if leaf - define a class label
        self.atrIDs = atrIDs                       #attributes this node is awair of
        self.instanceIDs = instanceIDs             #list of training instance ID's this node contains
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
            if data['data'][i][self.classCol] == data['attributes'][i][1][classValueID]:
                count += 1
        return count
        
    def Hd(self,positive,instances):  #assumes binary class label
        Py = positive / instances ;  Pyc = 1 - Py
        H = Py * math.log(Py , 2) - Pyc * math.log(Pyc , 2)
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
             condition - if atr is nominal, condition is the nominal value ID
                         if atr is numeric, condition is proposed threashold
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
                        if self.data['data'][ID][self.classCol] == data['attributes'][ID][1][0]:
                            positive += 1
                counts.append(count)
                positives.append(positive)
            
            #calculate weighted entropy per choice (I.E. conditional entropy on atrID)  [3.27]
            wHd = 0
            for i, choice in enumerate(choices):
                wHd += (counts[i]/self.nInstances) * self.Hd(positives[i],counts[i])
      
        
        #numeric type attributes
        if type(self.data['attributes'][atrID][1]) == unicode:
            #calculate training instances on either side of the threashold and positives on either side
            counts = [0,0]; positives = [0,0]  #[<=, >] 
            for ID in self.instanceIDs:
                if self.data['data'][ID][atrID] <= threashold:
                    counts[0] += 1
                    if self.data['data'][ID][self.classCol] == data['attributes'][ID][1][0]:
                         positives[0] += 1
                else:
                    counts[1] += 1 
                    if self.data['data'][ID][self.classCol] == data['attributes'][ID][1][0]:
                         positives[1] += 1
            #calculate weighted entropy for numerical attribute on this threashold:
            wHd = 0
            for i , condition in enumerate(['belowThreashold','aboveThreashold']):
                wHd += (counts[i]/self.nInstances) * self.Hd(positives[i],counts[i])
               
                    
            
           
    
    def InfoGain(self,atrID, threashold = False):
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
                Xi = np.zeros([2,self.nInstances])
                #create a lookup table for class labels (binary)
                classDict = {self.data['attributes'][self.classCol][1][0]:1 ,  #negative
                             self.data['attributes'][self.classCol][1][1]:0}   #positive
                #populate matrix
                for newID, instanceID in enumerate(self.instanceIDs):
                    Xi[newID,0] = self.data['data'][instanceID][candidateSplit]
                    Xi[newID,1] = classDict[self.data['data'][instanceID][self.classCol]]
                #perform argsort, get indecies that would sort the array
                sortIDs = np.argsort(Xi[0,:])
                #form groups of same value numerical features and class change indicators:
                #   0 == negative, 1 == positive, 2 = both in same group
                groups = []
                currentGroupLabel = Xi[sortIDs[0],0]
                currentGroup = [Xi[sortIDs[0],0],Xi[sortIDs[0],0]]
                for ID in sortIDs:
                    if Xi[ID,0] != currentGroupLabel:
                        groups.append(currentGroup)
                        currentGroupLabel = Xi[ID,0]
                        currentGroup = [Xi[ID,0],Xi[ID,1]]
                    if Xi[ID,0] == currentGroupLabel:
                        if currentGroup[1] != Xi[ID,1]:  #both groups in same class
                            currentGroup[1] = 2
                #form threasholds at the midpoint of disimilar groups
                threasholds = []
                for i in range(len(groups - 1)):
                    if groups[i][1] != groups[i+1][1]: #class boundry
                        threasholds.append((groups[i][0] + groups[i+1][0])/2)
                
                #append [candidate split , threashold] to list of candidate splits
                for threashold in threasholds:
                    candidateSplits.append([candidateSplit , threashold])
        return candidateSplits 
                    
    def stoppingCriteria(self):
        pass
    
    def FindBestSplit(self):
        pass 
    
    def makeSubtree(self):
        """
        starting from current node, make a subtree recursively
        """
        c = self.determineCandidateSplits()
        #check if stopping criteria met
        if self.stoppingCriteria():
            self.leaf = True
            self.classLabel = 0 ## this should be the most populous label?
        else:
            splitID = self.FindBestSplit() #feature ID
            # for each outcome k in S
            #make a node with subset of indicies and attributes
            #add node to list with append
        
        return
            
    
    def printTree(self):
        pass


#---------------------------- tests -------------------------------------------

class TestID3(unittest.TestCase):
    
    def testNode(self):
        pass


tests = True
if tests:
    unittest.main()