'''
Created on 2 avr. 2017

@author: Naitra
'''
from xlrd.formula import num2strg

from Maths.ClosedForm.BlackScholes import Call
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.random import multivariate_normal
from Scripts.CreditModel.Tools.RiskStatistics import risk_statistics
from statsmodels.distributions.empirical_distribution import ECDF

if __name__ == '__main__':
    pass

''' 
Parameters
'''
# parameters
rho = -0.1
correlation = np.array([[1, rho], [rho, 1]])
sigma = 0.2
r = 0.1
F0 = 100


# contract
T = 1

# portfolio
K = 100

Nexposure = 10
Ncalculus = 150
timeExposure = np.linspace(0, T, Nexposure, endpoint=True) # not necessary the point we need to price
calculationDate = np.linspace(0, T, Ncalculus, endpoint=True)

timeline = sorted(set(timeExposure) | set(calculationDate))
index_exposure = [timeline.index(t) for t in timeExposure]
Ntimes = len(timeline)
# Simulation
Nsimulations = 100
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

gbm = [(r-0.5*sigma**2)*timeline[t] + sigma*Paths[:, t, 0] for t in range(0, Ntimes)]
gbm = np.transpose(F0*np.exp(gbm))

priceCall = lambda s, t: Call(s, r, sigma, K, T-t)
Exposures = np.transpose([ [priceCall(pathtime_k, timeline[t]) for pathtime_k in gbm[:, t] ] for t in index_exposure ])

constant_prob_def = 0.001 # calibrated from cds
prob_def = [constant_prob_def for k in range(0, Nexposure) ]
norminv = norm.ppf
C = [norminv(prob_def[k]) for k in range(0, Nexposure)]

# market factor
ecdf_exposure = [ECDF(Exposures[:,t]) for t in range(0, Nexposure)]
Z = np.transpose([norminv(ecdf_exposure[t](Exposures[:,t])) for t in range(0, Nexposure)])
denom = np.sqrt(1-rho*rho)
weights = np.transpose([norm.cdf((C[t]-rho*Z[:, t])/denom) for t in range(0, Nexposure)])

weights = np.transpose([weights[:, t]/sum(weights[:, t]) for t in range(0, Nexposure)])

###################
### Compute stats
###################

alpha = 0.05

resultsIndep = np.array([risk_statistics(Exposures[:, t], alpha=alpha) for t in range(0, Nexposure)])
resultsWWR = np.array([risk_statistics(Exposures[:, t], weights=weights[:, t], alpha=alpha) for t in range(0, Nexposure)])
# first dimension is time

###################
### plots
###################


fig = plt.figure()

risk_measure = ['Expected Exposure','PFE'+num2strg(alpha),'ES'+num2strg(alpha)]
for k in range(len(risk_measure)): # 3 is the number of parameters
    ax = plt.subplot(len(risk_measure),1,k+1)
    plt.plot(timeExposure, resultsIndep[:,k],'blue') # EE
    plt.plot(timeExposure, resultsWWR[:,k],'red') # EE
    plt.title(risk_measure[k])
    #plt.xticks(x, tenors[t,:],rotation='vertical')
    plt.xlabel('dates')

#plt.title("Risk measure against time")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=3, fancybox=True, shadow=True)
plt.legend(['indep','WWR'])
plt.show()

plt.plot(timeExposure, np.transpose(Exposures), color='green', marker='o', markerfacecolor='None',linestyle = 'None')
plt.xlabel('time')
plt.ylabel('value')
plt.title('Exposure')
plt.show()

