'''
Created on 21 mars 2017

@author: Naitra
'''
from numpy.random import multivariate_normal
from numpy import zeros


class BrownianMotion(object):
    '''
    classdocs
    '''

    def __init__(self, covariance, mean=None):
        '''
        Constructor
        '''
        self.Covariance = covariance
        self.Dimension = len(self.Covariance[0])
        self.Mean = zeros(self.Dimension) if mean == None else mean
        
    def Path(self, timeline, nbSimulation):
        '''
        Create nbSimulation path of the brownian motion evaluated at timeline
        '''
        result = multivariate_normal(self.Mean,self.Covariance,(nbSimulation,len(timeline)))
        result[:,0,:] *= timeline[0]
        for k in range(1,len(timeline)):
            result[:,k,:] = (timeline[k]-timeline[k-1])*result[:,k,:] + result[:,k-1,:]
        return result
            
            
        