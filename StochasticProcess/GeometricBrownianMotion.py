'''
Created on 20 mars 2017

@author: Naitra
'''
from numpy import zeros,sqrt,power,array,exp,log
from numpy.matlib import randn
class GeometricBrownianMotion(object):
    '''
    classdocs
    '''


    def __init__(self, initialValue, drift,vol):
        '''
        Constructor
        '''
        self.InitialValue = initialValue
        self.Drift = drift
        self.Vol = vol
        
    def path(self,timeline,N):
        '''
        Create N path of the GBM evaluated at the values in timeline
        '''
        dsigmas = 1.0/2*power(self.Vol,2)
        adjusedDrift = (self.Drift-dsigmas)
        T = len(timeline)
        L = zeros([N,T])
        W = array(randn((N,T-1)))
        L[:,0]  = log(self.InitialValue) #broadcasting
        for j in range(1,T):
            delta = timeline[j]-timeline[j-1]
            L[:,j] = L[:,j-1] + adjusedDrift*delta + self.Vol*sqrt(delta)*W[:,j-1]
        L = exp(L)
        
    