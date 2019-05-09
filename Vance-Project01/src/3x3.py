# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 00:41:03 2018

@author: Shane
"""

import numpy as np
import pcn_logic_eg as ple

""" Training Set """

# The character I [3x3 matrix]
iOne = [1,1,1]+[0,1,0]+[1,1,1]
iTwo = [0,1,0]+[0,1,0]+[0,1,0]
iThree = [0,0,0]+[0,0,1]+[0,0,1]

# The character L [3x3 matrix]
lOne = [1,0,0]+[1,0,0]+[1,1,1]
lTwo = [1,0,0]+[1,0,0]+[1,0,0]
lThree = [0,1,0]+[0,1,0]+[0,1,1]

trainingSet = np.array([iOne, iTwo, iThree, lOne, lTwo, lThree])
targets = np.array([[0],[0],[0],[1],[1],[1]])

p = ple.pcn(trainingSet,targets)

# train the two characters
p.pcntrain(trainingSet,targets,1,100)


""" Test Set """

# print a new line
print("\nConfusion-Matrix: ")

# The character I [3x3 matrix]
iOneS = [0,0,1]+[0,0,1]+[0,0,1]
iTwoS = [1,1,1]+[0,1,0]+[1,1,0]
iThreeS = [1,0,1]+[1,1,1]+[0,0,1]

# The character L [3x3 matrix]
lOneS = [1,0,0]+[1,0,0]+[1,1,1]
lTwoS = [1,1,1]+[1,0,0]+[1,0,0]
lThreeS = [0,1,0]+[0,1,0]+[0,1,1]

testSet = np.array([iOneS, iTwoS, iThreeS, lOneS, lTwoS, lThreeS])
p.confmat(testSet, targets)