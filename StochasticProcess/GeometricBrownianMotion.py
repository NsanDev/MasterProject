'''
Created on 20 mars 2017

@author: Naitra
'''
from numpy import zeros, sqrt, array, exp, log
from numpy.matlib import randn
class GeometricBrownianMotion(object):
    '''
    classdocs
    '''

    '''
    Constructor
    '''
    def __init__(self, initial_value, drift,vol):

        self.InitialValue = initial_value
        self.Drift = drift
        self.Vol = vol

    '''
    Create nb_path path of the GBM evaluated at the values in timeline
    '''

    def Path(self, timeline, nb_path):

        dsigmas = 0.5*self.Vol**2
        adjused_drift = (self.Drift-dsigmas)
        T = len(timeline)
        L = zeros([nb_path, T])
        W = array(randn((nb_path, T)))
        L[:, 0] = log(self.InitialValue) + adjused_drift * timeline[0] + self.Vol * sqrt(timeline[0]) * W[:, 0]
        for t in range(1,T):
            delta = timeline[t]-timeline[t-1]
            L[:, t] = L[:, t-1] + adjused_drift*delta + self.Vol*sqrt(delta)*W[:, t]
        return exp(L)