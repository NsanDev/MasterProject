'''
Created on 2 avr. 2017

@author: Naitra
'''
from xlrd.formula import num2strg


from numpy import transpose ,sum ,array,linspace,sqrt
import matplotlib.pyplot as plt
from scipy.stats import norm

from Maths.ClosedForm.BlackScholes import Call,Put
from Scripts.CreditModel.Tools.RiskStatistics import risk_statistics
from statsmodels.distributions.empirical_distribution import ECDF
from StochasticProcess.GeometricBrownianMotion import GeometricBrownianMotion
if __name__ == '__main__':
    pass


###################
### Parameters
###################

# parameters gbm
rho = -0.1
sigma = 0.2
mu = 0.2
r = 0.1
F0 = 100

# contract
T = 1

# portfolio
K = 100

nb_exposure = 10
nb_point_path = 150
time_exposure = linspace(0, T, nb_exposure, endpoint=True) # not necessary the point we need to price
time_path = linspace(0, T, nb_point_path, endpoint=True)

timeline = sorted(set(time_exposure) | set(time_path))
index_exposure = [timeline.index(t) for t in time_exposure]
Ntimes = len(timeline)
# Simulation
Nsimulations = 100
# market sensitivities

gbm = GeometricBrownianMotion(F0, mu, sigma)
simulated_paths = gbm.Path(timeline=timeline, nb_path=Nsimulations)


###################
### Exposure
###################
price_call = lambda s, t: Call(s, r, sigma, K, T-t)
price_put = lambda s, t: Put(s, r, sigma, K, T-t)
portfolio = [price_call, price_put]
cumulated_price = lambda s, t: sum([x(s,t) for x in portfolio])

Exposures = [[cumulated_price(pathtime_k, timeline[t]) for pathtime_k in simulated_paths[:, t]] for t in index_exposure]


constant_prob_def = 0.001 # have to be calibrated from cds
prob_def = [constant_prob_def for k in range(0, nb_exposure)]

C = [norm.ppf(prob_def[k]) for k in range(0, nb_exposure)]

# market factor
ecdf_exposure = [ECDF(Exposures[t]) for t in range(0, nb_exposure)]
Z = [norm.ppf(ecdf_exposure[t](Exposures[t])) for t in range(0, nb_exposure)]
denom = sqrt(1-rho*rho)
weights = [norm.cdf((C[t]-rho*Z[t])/denom) for t in range(0, nb_exposure)]

weights = [weights[t] / sum(weights[t]) for t in range(0, nb_exposure)]

###################
### Compute stats
###################

alpha = 0.05

resultsIndep = array([risk_statistics(Exposures[t], alpha=alpha) for t in range(0, nb_exposure)])
resultsWWR = array([risk_statistics(Exposures[t], weights=weights[t], alpha=alpha) for t in range(0, nb_exposure)])
# first dimension is time

###################
### plots
###################

fig = plt.figure()

risk_measure = ['Expected Exposure','PFE'+num2strg(alpha),'PFE'+num2strg(1-alpha),'ES'+num2strg(alpha)]
for k in range(len(risk_measure)): # 3 is the number of parameters
    ax = plt.subplot(len(risk_measure)/2,2,k+1)
    plt.plot(time_exposure, resultsIndep[:, k], 'blue')
    plt.plot(time_exposure, resultsWWR[:, k], 'red')
    plt.title(risk_measure[k])
    plt.xlabel('dates')
    # plt.xticks(x, tenors[t,:],rotation='vertical')
plt.legend(['Indep','WWR'])

#plt.title("Risk measure against time")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=3, fancybox=True, shadow=True)
plt.show()

plt.plot(time_exposure, array(Exposures), color='green', marker='o', markerfacecolor='None', linestyle ='None')
plt.xlabel('time')
plt.ylabel('value')
plt.title('Exposure')
plt.show()

