"""
@file:neuralnet.py
@author: Alex Dawson-Elli
@purpose:run this script to train and output (to output.txt) cross validation results for 
a 3 layer NN implementation
"""

#---------------------------- imports -----------------------------------------
import BNT
import sys

#------------------- train and test decision trees ----------------------------
BNT.grading(sys.argv[1], sys.argv[2], sys.argv[3] )