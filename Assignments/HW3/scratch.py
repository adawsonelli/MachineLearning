"""
scratch.py - test out ideas, structures etc. 
"""


import numpy as np
import matplotlib.pyplot as plt

#plot the sigmoid function
x = np.linspace(-4,4,1000)
sig = 1/(1 + np.exp(-x))
#plt.plot(x,sig)

#plot the cross-entropy loss of a function
def crossEntropy(a,y):
    return -1*(y*np.log(a) + (1-y)*np.log(1-a)) 

y1 = 1
y2 = 0
a  = np.linspace(.01,.99,100)

ce1 = crossEntropy(a,y1)
ce2 = crossEntropy(a,y2)
  
plt.plot(a,ce1)
plt.plot(a,ce2)
 
"""
notes on assignment:
    -what is a bias unit? - 
    -what is n-fold stratified cross validation?
    -
    
notes from forums: 
    -
"""
