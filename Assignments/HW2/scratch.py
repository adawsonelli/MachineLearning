"""
Created on Sun Oct 01 17:47:39 2017
@file: scratch.py
@author: Alex Dawson-Elli
@purpose: test out new ideas about assignment 2
"""

#------------------------------ imports ---------------------------------------
import numpy as np
import math
from matplotlib import pyplot as plt

#------------plot the entropy function for a binary variable-------------------

y1 = np.linspace(.001,1.001,1000); #probability of y1
y2 = np.ones(1000) - y1   #1-probability of y1
H = - (y1 * np.log2(y1) + y2 * np.log2(y2)) #vectorized H calculation

plt.plot(y1,H) 

#--------------------------- fibo ---------------------------------------------

def fibo(n):
    """
    calculates the nth term in the fibinoci sequence
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    else:
        return fibo(n-1) + fibo(n-2)

#------------------ test out mulitple init functions --------------------------
#class InitTest:
#    def __init__(self,one,two):
#        self.one = 1
#        self.two = 2
#    def __init__(self,one):
#        two = 2
#        self.__init__(one,two)
    
#note, approach does not work, there doesn't seem to be a good approach to multiple 
#constructors in python besides default arguments