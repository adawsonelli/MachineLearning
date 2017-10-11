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
import math


#------------------------ load data -------------------------------------------
data = arff.load(open('weather.arff','rb'))


#-------------------------- classes -------------------------------------------

class Node:
    """
    node forms the basic unit of the a tree 
    """     
    def __init__(self, data , m , atrIDs = None , instanceIDs = None):
        #when initializing first top level node
        if atrIDs == None and instanceIDs == None: 
            atrIDs = range(len(data['data']))
            instanceIDs = range(len(data['data']))
            
        self.data = data                               #reference to system data dictionary
        self.m = m                                     #stopping criteria
        self.children = []                             #children nodes
        self.isLeaf = False
        self.classLabel = None                         #if leaf - define a class label
        self.atrIDs = atrIDs                           #attributes this node is awair of
        self.instanceIDs = instanceIDs                 #list of training instance ID's this node contains
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
            if data['data'][i][self.classCol] == data['attributes'][self.classCol][1][classValueID]:
                count += 1
        return count
        
    def Hd(self,positive,instances):  #assumes binary class label
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
                        if self.data['data'][ID][self.classCol] == data['attributes'][self.classCol][1][0]:
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
                    if self.data['data'][ID][self.classCol] == data['attributes'][self.classCol][1][0]:
                         positives[0] += 1
                else:
                    counts[1] += 1 
                    if self.data['data'][ID][self.classCol] == data['attributes'][self.classCol][1][0]:
                         positives[1] += 1
            #calculate weighted entropy for numerical attribute on this threashold:
            wHd = 0
            for i , condition in enumerate(['belowThreashold','aboveThreashold']):
                wHd += (counts[i]/self.nInstances) * self.Hd(positives[i],counts[i])
               
                    
            
           
    
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
                    
    def stoppingCriteria(self,candidateSplits):
        """
        stopping criteria for making a leaf node are:
            (i)   all training instances belonging to the node are the same class
            (ii)  there are fewer than m training instances reaching the node
            (iii) no feature has positive info gain
            (iv)  there are no more remaining candidate splits
        
        return: True if should form a leaf, return false otherwise.
        """
        #(i) all training instances belonging to the node are the same class
        positiveFlag = 0; negativeFlag = 0
        for ID in self.instanceIDs:
            if self.data['data'][ID][self.classCol] == data['attributes'][self.classCol][1][0]:
                positiveFlag = 1;
            else:
                negativeFlag = 1;
        if not (positiveFlag and negativeFlag):
            return True
        
        #(ii)  there are fewer than m training instances reaching the node
        if len(self.instanceIDs) < self.m:
            return True
        
        #(iii) no feature has positive info gain
        infoGainList = []
        for split in candidateSplits:
            #nominal feature
            if type(split == int):
                infoGainList.append(self.infoGain(split))
            #numeric feature
            if type(split == list):
                infoGainList.append(self.infoGain(split[0],split[1]))
        
        #(iv)  there are no more remaining candidate splits
        if len(self.atrIDs) == 0 or len(candidateSplits) == 0:
            return True
        
        #none of the stopping criteria have been met
        return False
    
    def findClassLabel(self,parent):
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
        elif self.pos == self.neg:  #this doesn't handle if the parent class also is the same
            if parent.pos > parent.neg:
                return parent.data['attributes'][parent.classCol][1][0]
            elif parent.pos < parent.neg:
                return parent.data['attributes'][parent.classCol][1][1]
            

    def findBestSplit(self, candidateSplits):
        """
        score each of the candidate splits in terms of info gain. and return best
        returns:
            if nominal:atrID of highest info gain 
            if numeric:[atrID, threashold]
        """
        bestSplit  = {} ; bestSplit['infoGain'] = -1000
        for cs in candidateSplits:
            #nominal
            if type(cs) == unicode:
                infoGain = self.infoGain(cs)
            #numeric
            if type(cs) == list:
                infoGain = self.infoGain(cs[0],cs[1])
            
            #handle infogain
            if infoGain > bestSplit['infoGain']:
                bestSplit['ID'] = cs
                bestSplit['infoGain'] = infoGain
        
        return bestSplit['ID']
    
    
    def makeSubtree(self, parent = None):   #[3.9]
        """
        starting from current node, make a subtree recursively
        """
        candidateSplits = self.determineCandidateSplits()
        
        #check if stopping criteria met
        if self.stoppingCriteria(candidateSplits):
            self.leaf = True
            self.classLabel = self.findClassLabel(parent)
        else:
            split = self.FindBestSplit(candidateSplits) #feature ID
            
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
                split[0] = splitID ; threashold = split[1]
                sortedInstances = [[],[]]    #[[lower],[upper]]
                for ID in self.instanceIDs:
                    if self.data['data'][ID][splitID]   <= threashold:
                        sortedInstances[0].append(ID)
                    elif self.data['data'][ID][splitID]  < threashold:
                        sortedInstances[1].append(ID)
            
            
            #for each outcome k in S - make a new node and add to children
            childAtrIDs = self.atrIDs.remove(splitID) #can't split on this feature again (is this true for numerics?!)
            for childInstanceIDs in sortedInstances:
                childNode = Node(self.data,self.m,childAtrIDs,childInstanceIDs)
                self.children.append(childNode)
            
            #for each child, make a subtree
            for child in self.children:
                child.makeSubtree(self)  #pass in self as parent
        
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