'''
Created on 2 avr. 2017

@author: Naitra
'''

import matplotlib.pyplot as plt
from numpy import exp
from numpy import transpose, sum, array, linspace, vectorize, random, maximum
from xlrd.formula import num2strg

from CreditModel.DirectionalWayRisk.Weights import Merton
from CreditModel.Tools.RiskStatistics import risk_statistics
from Maths.ClosedForm.BlackScholes import Call, Put
from Maths.PiecewiseFlat import piecewise_flat
from StochasticProcess.GeometricBrownianMotion import GeometricBrownianMotion

if __name__ == '__main__':
    pass


###################
### Parameters
###################

# parameters gbm
sigma = 0.2
mu = 0.2
r = 0.1
F0 = 45

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

nb_point_exposure = 10
nb_point_path = 20
nb_point_intensity = 5
start_exposure = 0.05
start_default = 0.1
start_path = 0.01
range_exposure = range(0,nb_point_exposure)

time_exposure = linspace(start_exposure, T, nb_point_exposure, endpoint=True)  # times at which we want exposure
time_intensity = linspace(start_exposure, T, nb_point_intensity, endpoint=True) # times where calibrated intensity breaks
time_path = linspace(start_path, T, nb_point_path, endpoint=True)

timeline = sorted(set(time_exposure) | set(time_path) | set(time_intensity))
index_exposure = [timeline.index(t) for t in time_exposure]
Ntimes = len(timeline)

###################
### Simulation
###################

# Simulation
Nsimulations = 100
random.seed(128)
# market sensitivities
gbm = GeometricBrownianMotion(F0, mu, sigma)
simulated_paths = gbm.Path(timeline=timeline, nb_path=Nsimulations)
simulated_paths = transpose(simulated_paths,(1,0))

#model = Schwartz97(S0=S0,delta0=delta0,mu=mu,sigma_s=sigma_s,kappa=kappa,alpha=alpha,sigma_e=sigma_e,rho=rho)
#simulated_paths = model.Path(timeline,nb_path=Nsimulations,spot_only=True)

###################
### Portfolio
###################

# portfolio
K = 45

price_call = lambda s, t: Call(s, r, sigma, K, T-t)
price_put = lambda s, t: Put(s, r, sigma, K, T-t)
price_spot = lambda s, t: s
#portfolio = [price_call, price_put]
portfolio = [price_spot,price_put,price_call]
portfolio = [vectorize(instr) for instr in portfolio]
cumulated_price = lambda s, t: sum([instrument(s, t) for instrument in portfolio], axis=0)
nb_instrument = len(portfolio)
range_portfolio = range(0, nb_instrument)
V = [[instrument(simulated_paths[t,:], timeline[t]) for t in index_exposure] for instrument in portfolio ]
#Vtot = [cumulated_price(simulated_paths[t,:], timeline[t]) for t in index_exposure]
###################
### Exposure
###################

#start = time.time()
Collateral_level = 0
Exposure_function = lambda V:  maximum(V-Collateral_level,0)
Exposures = [[Exposure_function(V[k][t]) for t in range_exposure] for k in range_portfolio]
#Exposurestot = [Exposure_function(Vtot[t]) for t in range_exposure]
#end = time.time()
#deltaT2 = end-start


###################
### Compute probabilities
###################

constant_intensity = 0.001 # have to be calibrated from cds
probability_default_value = 1 - exp(-constant_intensity*time_intensity)
PD = [piecewise_flat(t,probability_default_value,time_intensity) for t in time_exposure]

###################
### Compute weights
###################
rho = 0.5
weightsMerton = [[Merton(Exposures[k][t], rho, PD[t], tolerance=0.001) for t in range_exposure] for k in
                 range_portfolio]

###################
### Compute stats
###################

confidence = 0.05

resultsIndep = array(
    [[risk_statistics(Exposures[k][t], alpha=confidence) for t in range_exposure] for k in range_portfolio])
resultsDWR = array(
    [[risk_statistics(Exposures[k][t], weights=weightsMerton[k][t], alpha=confidence) for t in range_exposure] for k in
     range_portfolio])
#resultsDWRtot = array([risk_statistics(Exposures[t], weights=weightsMerton[t], alpha=alpha) for t in range_exposure])
# first dimension is time

###################
### CVA
###################

DiscountFactorXdefault = exp(-r*time_exposure)*PD
CVA_Indep = [sum(resultsIndep[k,:,0]*DiscountFactorXdefault) for k in range_portfolio]
CVA_DWR = [sum(resultsDWR[k,:,0]*DiscountFactorXdefault) for k in range_portfolio]

###################
### plots
###################

fig = plt.figure()

risk_measure = ['Expected Exposure','PFE'+num2strg(alpha),'PFE'+num2strg(1-alpha),'ES'+num2strg(alpha)]
for k in range(len(risk_measure)): # 3 is the number of parameters
    ax = plt.subplot(len(risk_measure)/2,2,k+1)
    plt.plot(time_exposure, resultsIndep[0,:, k], 'blue',label='Indep')
    plt.plot(time_exposure, resultsDWR[0,:, k], 'red',label='DRW')
    plt.title(risk_measure[k])
    plt.xlabel('dates')
    plt.legend()
    # plt.xticks(x, tenors[t,:],rotation='vertical')
#plt.figlegend( fig, ['Indep', 'WWR'], 'upper right' )

#plt.title("Risk measure against time")
#fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=3, fancybox=True, shadow=True)
#fig.legend(['Indep','DWR'])
plt.show()

#plt.plot(time_exposure, array(Exposures), color='green', marker='o', markerfacecolor='None', linestyle ='None')
