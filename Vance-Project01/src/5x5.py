# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 00:41:03 2018

@author: Shane
"""

import numpy as np
import pcn_logic_eg as ple

""" Training Set """

# The character I [5x5 matrix]
iOne = [1,1,1,1,1]+[0,0,1,0,0]+[0,0,1,0,0]+[0,0,1,0,0]+[1,1,1,1,1]
iTwo = [0,0,1,0,0]+[0,0,1,0,0]+[0,0,1,0,0]+[0,0,1,0,0]+[0,0,1,0,0]
iThree = [1,0,1,0,1]+[0,0,1,0,0]+[0,0,1,0,0]+[0,0,1,0,0]+[1,0,1,0,1]
iFour = [1,1,1,1,1]+[0,1,1,1,0]+[0,1,1,1,0]+[0,1,1,1,0]+[1,1,1,1,1]

# The character L [5x5 matrix]
lOne = [1,1,0,0,0]+[1,1,0,0,0]+[1,1,0,0,0]+[1,1,1,1,1]+[1,1,1,1,1]
lTwo = [0,0,1,1,0]+[0,0,1,1,0]+[0,0,1,1,0]+[0,0,1,1,0]+[0,0,1,1,0]
lThree = [1,0,0,0,0]+[1,0,0,0,0]+[1,0,0,0,0]+[1,0,0,0,0]+[1,1,1,1,1]
lFour = [0,1,0,0,0]+[0,1,0,0,0]+[0,1,0,0,0]+[0,1,0,0,0]+[0,1,0,0,0]

trainingSet = np.array([iOne, iTwo, iThree, iFour, lOne, lTwo, lThree, lFour])
targets = np.array([[0],[0],[0],[0],[1],[1],[1],[1]])

p = ple.pcn(trainingSet,targets)

# train the two characters
p.pcntrain(trainingSet,targets,1,100)


""" Test Set """

# print a new line
print("\nConfusion-Matrix: ")
# The character I [5x5 matrix]
iOneS = [1,1,1,1,1]+[0,0,1,1,0]+[0,0,1,0,0]+[0,0,1,0,0]+[1,1,1,1,1]
iTwoS = [1,1,1,0,0]+[1,1,1,0,0]+[1,1,1,1,0]+[0,0,1,0,0]+[1,1,1,0,0]
iThreeS = [1,0,1,0,1]+[0,0,1,0,0]+[0,0,1,0,0]+[0,0,1,0,0]+[1,0,1,0,1]
iFourS = [1,1,1,1,1]+[0,1,1,1,0]+[0,1,1,1,0]+[0,1,1,1,0]+[1,1,1,1,1]

# The character L [5x5 matrix]
lOneS = [1,1,0,0,0]+[1,1,0,0,0]+[1,1,0,0,0]+[1,1,1,1,1]+[1,1,1,1,1]
lTwoS = [0,0,1,1,0]+[0,0,1,1,0]+[0,0,1,1,0]+[0,0,1,1,0]+[0,0,1,1,0]
lThreeS = [1,0,0,0,0]+[1,0,0,0,0]+[1,0,0,0,0]+[1,0,0,0,0]+[1,1,1,1,1]
lFourS = [0,1,0,0,0]+[0,1,0,0,0]+[0,1,0,0,0]+[0,1,0,0,0]+[0,1,0,0,0]

testSet = np.array([iOneS, iTwoS, iThreeS, iFourS, lOneS, lTwoS, lThreeS, lFourS])
p.confmat(testSet, targets)