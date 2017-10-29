"""
scratch.py - test out ideas, structures etc. 
"""


import numpy as np
import matplotlib.pyplot as plt

#plot the sigmoid function
x = np.linspace(-4,4,1000)
sig = 1/(1 + np.exp(-x))
plt.plot(x,sig)

 
"""
notes on assignment:
    -what is a bias unit? - 
    -what is n-fold stratified cross validation?
    -
    
notes from forums: 
    -
"""
