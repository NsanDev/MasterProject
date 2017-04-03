'''
Created on 2 avr. 2017

@author: Naitra
'''
import StochasticProcess.Multidimensional.BrownianMotion as BrownianMotion
from StochasticProcess.GeometricBrownianMotion import GeometricBrownianMotion as GBM
from Maths.ClosedForm.BlackScholes import Call
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pass

''' 
Parameters
'''
# parameters
rho = 0.1
correlation = np.array([[1,rho],[rho,1]])
sigma = 0.2
r = 0.1
F0 = 100
# contract
T = 1

# portfolio
K = 100

N = 100
timeExposure = np.linspace(0, T, N, endpoint=True) # not necessary the point we need to price

timeline = timeExposure
Ntimes = len(timeline)
#Simulation
Nsimulations = 15
'''
Create Paths
'''
gbm = GBM(F0,r,sigma)
simulations = gbm.Path(timeExposure,Nsimulations)

priceCall = lambda s,t: Call(s,r,sigma,K,T-t)
Exposures = np.array([ [priceCall(pathtime_k,timeline[k]) for pathtime_k in simulations[:,k] ]  for k in range(0,Ntimes)])

plt.plot(timeline,Exposures)
plt.xlabel('time')
plt.ylabel('value')
plt.title('Exposure')
plt.show()