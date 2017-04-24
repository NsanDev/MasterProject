'''
Created on 2 avr. 2017

@author: Naitra
'''
from xlrd.formula import num2strg


from numpy import transpose ,sum ,array,linspace,sqrt,vectorize
import matplotlib.pyplot as plt
from scipy.stats import norm

from Maths.ClosedForm.BlackScholes import Call,Put
from Scripts.CreditModel.Tools.RiskStatistics import risk_statistics
from statsmodels.distributions.empirical_distribution import ECDF
from StochasticProcess.GeometricBrownianMotion import GeometricBrownianMotion
from StochasticProcess.Commodities.Schwartz97 import Schwartz97
import time
if __name__ == '__main__':
    pass


###################
### Parameters
###################

# parameters gbm
rho = 0.2
sigma = 0.2
mu = 0.2
r = 0.1
F0 = 100

# parameters schwartz
S0 = F0
delta0 = 0.1
mu = 0.142
sigma_s = 0.393
kappa = 1.876
alpha = 0.393
sigma_e = 0.527
corr = 0.766
lamb = 0.198

# contract
T = 1

nb_exposure = 10
nb_point_path = 10
start_simulation = 0.01
start_exposure = 0.1

time_exposure = linspace(start_exposure, T, nb_exposure, endpoint=True) # not necessary the point we need to price
time_path = linspace(start_simulation, T, nb_point_path, endpoint=True)

timeline = sorted(set(time_exposure) | set(time_path))
index_exposure = [timeline.index(t) for t in time_exposure]
Ntimes = len(timeline)

# Simulation
Nsimulations = 100

# market sensitivities
gbm = GeometricBrownianMotion(F0, mu, sigma)
simulated_paths = gbm.Path(timeline=timeline, nb_path=Nsimulations)
simulated_paths = transpose(simulated_paths,(1,0))

#model = Schwartz97(S0=S0,delta0=delta0,mu=mu,sigma_s=sigma_s,kappa=kappa,alpha=alpha,sigma_e=sigma_e,rho=rho)
#simulated_paths = model.Path(timeline,nb_path=Nsimulations,spot_only=True)

###################
### Exposure
###################

# portfolio
K = 100

price_call = lambda s, t: Call(s, r, sigma, K, T-t)
price_put = lambda s, t: Put(s, r, sigma, K, T-t)
price_spot = lambda s, t: s
#portfolio = [price_call, price_put]
portfolio = [price_spot]
cumulated_price = lambda s, t: sum([vectorize(instrument)(s, t) for instrument in portfolio], axis=0)

start = time.time()
Exposures1 = [[cumulated_price(pathtime_k, timeline[t]) for pathtime_k in simulated_paths[t,:]] for t in index_exposure]
end = time.time()
deltaT = end-start
start = time.time()
Exposures = [list(cumulated_price(simulated_paths[t,:], timeline[t])) for t in index_exposure]
end = time.time()
deltaT2 = end-start
start = time.time()
Exposures3 = cumulated_price(simulated_paths, timeline)
end = time.time()
deltaT3 = end-start

constant_prob_def = 0.001 # have to be calibrated from cds
prob_def = [constant_prob_def for k in range(0, nb_exposure)]

C = [norm.ppf(prob_def[k]) for k in range(0, nb_exposure)]

# market factor
ecdf_exposure = [ECDF(Exposures[t]) for t in range(0, nb_exposure)]
def set_inf_to_mean(x,tolerance=0.001):
    if x == float('Inf'):
        return 1-tolerance
    elif x == -float('Inf'):
        return tolerance
    else: return x
set_inf_to_mean = vectorize(set_inf_to_mean)
Z = [set_inf_to_mean(norm.ppf(ecdf_exposure[t](Exposures[t]))) for t in range(0, nb_exposure)]

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
    plt.legend()
    # plt.xticks(x, tenors[t,:],rotation='vertical')
#plt.figlegend( fig, ['Indep', 'WWR'], 'upper right' )

#plt.title("Risk measure against time")
#fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=3, fancybox=True, shadow=True)
#fig.legend(['Indep','WWR'])
plt.show()

plt.plot(time_exposure, array(Exposures), color='green', marker='o', markerfacecolor='None', linestyle ='None')
plt.xlabel('time')
plt.ylabel('value')
plt.title('Exposure')
plt.show()