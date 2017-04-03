'''
Created on 21 mars 2017

@author: Naitra
'''
import unittest
import numpy as np
from StochasticProcess.GeometricBrownianMotion import  GeometricBrownianMotion as GBM
from StochasticProcess.Multidimensional import BrownianMotion

from numpy import linspace
from matplotlib import pyplot

class Test(unittest.TestCase):

    def testGeometricBrownianMotion(self):
        S0 = 100
        drift = 0.1
        vol = 0.1
        T = 1
        gbm = GBM(S0,drift,vol)
        timeline = linspace(0, T, 100, endpoint=True)
        nbSimulations = 100
        paths = gbm.Path(timeline,nbSimulations)
        pyplot.figure()
        for j in range(0,nbSimulations):
            pyplot.plot(timeline,paths[j,:])
            pyplot.ylabel('value')
            pyplot.xlabel("time")
        pyplot.show()
        pass

    def testBrownianMotion(self):
        correlation = np.array([[1,0.1,0.1],[0.1,1,0.1],[0.1,0.1,1]])
        bm = BrownianMotion(correlation)
        (tMin,tMax) = (0,1) 
        nbpointsTime = 100
        nbSimulations = 4 # choose even number
        
        timeline = linspace(tMin,tMax,nbpointsTime,endpoint = True)
        paths =  bm.Path(timeline, nbSimulations)
        
        pyplot.figure()
        for j in range(0,nbSimulations):
            pyplot.subplot(nbSimulations/2,2,j+1)
            #for i in range(0,bm.Dimension):
            pyplot.plot(timeline,paths[j,:,:])
            pyplot.ylabel('value')
            pyplot.xlabel("time")
        pyplot.show()
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testBrownianMotion']
    unittest.main()