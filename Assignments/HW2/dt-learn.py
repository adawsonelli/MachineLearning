"""
@file:dt-learn.py
@author: Alex Dawson-Elli
@purpose:run this script to train and classify with ID3 decision tree implementation
"""

#---------------------------- imports -----------------------------------------
import decisionTree
import sys

#------------------- train and test decision trees ----------------------------
decisionTree.outputTestResults(sys.argv[1], sys.argv[2], int(sys.argv[3]))