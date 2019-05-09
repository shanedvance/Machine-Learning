# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:01:04 2018

@author: Shane
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May  2 15:14:15 2018

@author: Shane
"""

# Shane Vance, 2018

# The sinewave regression example

import pylab as pl
import numpy as np

# Set up the data
x = np.linspace(0, np.pi / 2,100).reshape((100,1))
t = (np.sin(x)) + np.random.randn(100).reshape((100,1))*0.02

# Split into training, testing, and validation sets
train = x[0::2,:]
test = x[1::4,:]
valid = x[3::4,:]
traintarget = t[0::2,:]
testtarget = t[1::4,:]
validtarget = t[3::4,:]

# Plot the data
pl.plot(x,t,'o')
pl.xlabel('x')
pl.ylabel('t')

# Perform basic training with a small MLP
import mlp

# Test out different sizes of network
count = 0
out = np.zeros((10,7))
for nnodes in [1,2,3,5,10,25,50]:
    for i in range(10):
        net = mlp.mlp(train,traintarget,nnodes,outtype='logistic')
        net.mlptrain(train,traintarget,0.1,101)
        out[i,count] = net.earlystopping(train,traintarget,valid,validtarget,0.1)
    count += 1
    
results = np.concatenate((test,-np.ones((np.shape(test)[0],1))),axis=1)
outputs = net.mlpfwd(results)

#pl.plot(test,outputs,'x')
#pl.legend(('Training', 'Test'))

print("Error=", 0.5*np.sum((outputs-testtarget)**2))
print("Output=", out)
print("Mean=", out.mean(axis=0))
print("Std. dev=", out.var(axis=0))
print("Max=", out.max(axis=0))
print("Min=", out.min(axis=0))

#net.confmat(test, testtarget)
pl.show()