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
            Choices = self.data['attributes'][atrID][1]
            nChoices = len(choices)
            
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
            pass
            
           
    
    def InfoGain(self,atrID, threashold = False):
        """
        calculates the info gain of an attribute (specified by attribute ID) with
        respect to the node's instances
        """
        return self.entropy() - self.conditionalEntropy(atrID,threashold)
        
    
    def determineCandidateSplits(self):
        pass

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