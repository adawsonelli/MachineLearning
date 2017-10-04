# -*- coding: utf-8 -*-
"""
ID3
"""

#-------------------------- imports -------------------------------------------
import arff
#import numpy as np
import math


#------------------------ load data -------------------------------------------
data = arff.load(open('weather.arff','rb'))



#-------------------------- classes -------------------------------------------
#class data:
#    def __init__(self,arffDataDict):
#        self.data = arffDataDict['data']                 #list of lists
#        self.attributes = arffDataDict['attributes']     #
#
#



class node:
    """
    node forms the basic unit of the a tree 
    """
    def __init__(self, data , atrIDs , instanceIDs ):
        self.data
        self.children = []                         #children nodes
        self.isLeaf = False            
        self.atrIDs = atrIDs                       #attributes this node is awair of
        self.instanceIDs = instanceIDs             #list of training instance ID's this node contains
        self.nInstances = len(self.instanceIDs)
        self.pos = self.count('pos')             
        self.neg = self.count('negative')
    
    def count(self,label):
        """
        deterine the number of negative or positive instances in the set
        """
        classID = len(data['data'][0]) - 1  #last column
        if label == 'pos': classValueID = 0
        if label == 'negative': classValueID = 1
        
        count = 0
        for i in self.instanceIDs:
            if data['data'][i][classID] == data['attributes'][classID][1][classValueID]:
                count += 1
        return count
        
    
    def entropy(self):
        """
        calculates entropy of class label outcomes at this node
        """
        #assumes binary class label
        Py = self.pos / self.nInstances ;  Pyc = 1 - Py
        H = Py * math.log(Py , 2) - Pyc * math.log(Pyc , 2)
        return H
    
     def conditionalEntropy(self,atrID,condition):
        """
        calculates conditional entropy of attribution give the conditions
        inputs:
             atrID - integer ID of the atribute to be evaluated
             condition - if atr is nominal, condition is the nominal value ID
                         if atr is numeric, condition is proposed threashold
        """
        # this needs to be able to handle numeric features??!!??
        #should condition be a value, or an index? 
           
        
    def InfoGain(self):
        pass
    
    def determineCandidateSplits(self):
        pass
    
    def FindBestSplit(self):
        pass 
    
    def makeSubtree(self):
        pass
    
    def printTree(self):


#---------------------------- tests -------------------------------------------

class TestID3(unittest.TestCase):
    
    def testNode(self):
        pass


tests = True
if tests:
    unittest.main()