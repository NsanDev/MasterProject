'''
Created on 2 avr. 2017

@author: Naitra
'''

from Maths.ClosedForm.BlackScholes import Call
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.random import multivariate_normal
from Scripts.CreditModel.Tools.RiskStatistics import riskStatistics

if __name__ == '__main__':
    pass

''' 
Parameters
'''
# parameters
rho = 0.1
correlation = np.array([[1, rho], [rho, 1]])
sigma = 0.2
r = 0.1
F0 = 100


# contract
T = 1

# portfolio
K = 100

Nexposure = 100
Ncalculus = 150
timeExposure = np.linspace(0, T, Nexposure, endpoint=True) # not necessary the point we need to price
calculationDate = np.linspace(0, T, Ncalculus, endpoint=True)

timeline = sorted(set(timeExposure) | set(timeExposure))

Ntimes = len(timeline)
# Simulation
Nsimulations = 15
# market sensitivities

def Path(timeline, nb_simulation):
    '''
    Create nbSimulation path of the brownian motion evaluated at timeline
    '''
    result = multivariate_normal([0,0], correlation, (nb_simulation, len(timeline)))
    result[:, 0, 0] *= timeline[0]
    for k in range(1, len(timeline)):
        result[:, k, 0] = (timeline[k] - timeline[k - 1]) * result[:, k, 0] + result[:, k - 1, 0]
    return result

Paths = Path(timeline=timeline, nb_simulation=Nsimulations)

gbm = [ (r-0.5*sigma**2)*timeline[t] + sigma*Paths[:, t, 0] for t in range(0, Ntimes)]
gbm = np.transpose(F0*np.exp(gbm))

priceCall = lambda s, t: Call(s, r, sigma, K, T-t)
Exposures = np.transpose([ [priceCall(pathtime_k, timeline[t]) for pathtime_k in gbm[:, t] ] for t in range(0, Ntimes) ])

constant_prob_def = 0.01 # calibrated from cds
prob_def = [constant_prob_def for k in range(0, Ntimes) ]
norminv = norm.ppf
C = [norminv(prob_def[k]) for k in range(0, Ntimes)]
Z = norm.cdf(Exposures)# market factor
denom = np.sqrt(1-rho*rho)
weights = np.transpose([norm.cdf((C[t]-rho*Z[:, t])/denom) for t in range(0, Ntimes)])
#weights = np.array([weights[:, t]/sum(weights[:, t]) for t in range(0, Ntimes)])

###################
### Compute stats
###################
weighted_exposures = np.multiply(weights, Exposures)
alpha = 0.05

resultsIndep = [riskStatistics(Exposures[:, t], alpha) for t in range(0, Ntimes)]
resultsWWR = [riskStatistics(Exposures[:, t], weights[:, t], alpha) for t in range(0, Ntimes)]

plt.plot(timeline, Exposures)
plt.xlabel('time')
plt.ylabel('value')
plt.title('Exposure')
plt.show()