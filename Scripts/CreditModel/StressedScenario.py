'''
Created on 2 avr. 2017

@author: Naitra
'''

from numpy import transpose ,sum ,array,linspace,random, vstack
import matplotlib.pyplot as plt
from Maths.ClosedForm.BlackScholes import Call,Put
from Scripts.CreditModel.Tools.RiskStatistics import risk_statistics
from StochasticProcess.GeometricBrownianMotion import GeometricBrownianMotion
from StochasticProcess.Commodities.Schwartz97 import Schwartz97
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
F0 = 50
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

nb_exposure = 3
nb_point_path = 10
start_simulation = 0.01
start_exposure = 0.5

time_exposure = linspace(start_exposure, T, nb_exposure, endpoint=True) # not necessary the point we need to price
time_path = linspace(start_simulation, T, nb_point_path, endpoint=True)

timeline = sorted(set(time_exposure) | set(time_path))
index_exposure = [timeline.index(t) for t in time_exposure]
Ntimes = len(timeline)

# Simulation
Nsimulations = 1000
random.seed(128)
# market sensitivities
gbm = GeometricBrownianMotion(F0, mu, sigma)
# simulated_paths = gbm.Path(timeline=timeline, nb_path=Nsimulations)

model = Schwartz97(S0=F0,delta0=delta0,mu=mu,sigma_s=sigma_s,kappa=kappa,alpha=alpha,sigma_e=sigma_e,rho=rho)
simulated_paths = model.Path(timeline,nb_path=Nsimulations,spot_only=True)
###################
### Exposure
###################

# portfolio
K = 50

price_call = lambda s, t: Call(s, r, sigma, K, T-t)
price_put = lambda s, t: Put(s, r, sigma, K, T-t)
price_spot = lambda s, t: s
#portfolio = [price_call, price_put]
portfolio = [price_call,price_put]

cumulated_price = lambda s, t: sum([x(s,t) for x in portfolio])

#collateralized_price = lambda V: max(V-collateral)

probability_shock_cond_default = 0.5
magnitude = -0.1
std = 0.001

def shock(pshock,magnitude,std):
    if random.uniform()<pshock:
        return random.normal(1+magnitude,std)
    else: return 1
adjusted_shocks = array([shock(pshock=probability_shock_cond_default,magnitude=magnitude,std=std) for k in range(0,Nsimulations)])

Exposures = [[cumulated_price(pathtime_k, timeline[t]) for pathtime_k in simulated_paths[:, t]] for t in index_exposure]

ExposuresWWR = [[cumulated_price(adjusted_shocks[k]*simulated_paths[k,t], timeline[t]) for k in range(0,Nsimulations)] for t in index_exposure]

#ExposuresWWR =

###################
### Compute stats
###################

alpha = 0.05

resultsIndep = array([risk_statistics(Exposures[t], alpha=alpha) for t in range(0, nb_exposure)])
resultsDWR = array([risk_statistics(ExposuresWWR[t], alpha=alpha) for t in range(0, nb_exposure)])
# first dimension is time

###################
### plots
###################

'''
risk_measure = ['Expected Exposure','PFE'+num2strg(alpha),'PFE'+num2strg(1-alpha),'ES'+num2strg(alpha)]
for k in range(len(risk_measure)): # 3 is the number of parameters
    ax = plt.subplot(len(risk_measure)/2,2,k+1)
    plt.plot(time_exposure, resultsIndep[:, k], 'blue')
    plt.plot(time_exposure, resultsWWR[:, k], 'red')
    plt.title(risk_measure[k])
    plt.xlabel('dates')
    #plt.legend(['Indep', 'WWR'])
    # plt.xticks(x, tenors[t,:],rotation='vertical')
#plt.figlegend( fig, ['Indep', 'WWR'], 'upper right' )
'''
#plt.title("Risk measure against time")
#fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=3, fancybox=True, shadow=True)
#fig.legend(['Indep','WWR'])
bins = linspace(0, 50, 50)

'''
plt.hist(Exposures[0], bins, alpha=0.5, label='Indep')
plt.hist(ExposuresWWR[0], bins, alpha=0.5, label='WWR')
plt.title("Empirical Distribution of exposure at " + num2strg(timeline[0]) )
plt.xlabel("frequency")
plt.ylabel("exposure")
plt.legend(loc='upper right')
plt.show()
'''
data = vstack([Exposures[0], ExposuresWWR[0]]).T
plt.hist(data, bins, alpha=0.7, label=['Indep', 'DWR'])
plt.legend(loc='upper right')
plt.show()