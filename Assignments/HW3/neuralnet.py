"""
@file:neuralnet.py
@author: Alex Dawson-Elli
@purpose:run this script to train and output (to output.txt) cross validation results for 
a 3 layer NN implementation
"""

#---------------------------- imports -----------------------------------------
import NN
import sys

#------------------- train and test decision trees ----------------------------
NN.grading(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))