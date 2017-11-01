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
    
    #place transformed data into np.array dataMat
    nInstances  =  len(data['data'])
    nAttributes = len(data['data'][0])
    dataMat = np.zeros([nInstances,nAttributes])
    
    for InstID, instance in enumerate(data['data']):
        for atrID, attribute in enumerate(instance):
            #use dictionary to apply transformation
            dataMat[InstID,atrID] = atrLU[atrID][attribute] 
            
    return dataMat , data['attributes']
                    
#---------------------------- plotting ----------------------------------------

#---------------------------- grading -----------------------------------------
#---------------------------- debugging ---------------------------------------
