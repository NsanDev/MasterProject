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


    def __init__(self, initial_value, drift,vol):
        '''
        Constructor
        '''
        self.InitialValue = initial_value
        self.Drift = drift
        self.Vol = vol
        
    def Path(self, timeline, nb_path):
        '''
        Create Phi path of the GBM evaluated at the values in timeline
        '''
        dsigmas = 1.0/2*power(self.Vol, 2)
        adjused_drift = (self.Drift-dsigmas)
        T = len(timeline)
        L = zeros([nb_path, T])
        W = array(randn((nb_path, T - 1)))
        L[:, 0] = log(self.InitialValue)  # broadcasting
        for j in range(1,T):
            delta = timeline[j]-timeline[j-1]
            L[:, j] = L[:, j-1] + adjused_drift*delta + self.Vol*sqrt(delta)*W[:, j-1]
        return exp(L)
        
    