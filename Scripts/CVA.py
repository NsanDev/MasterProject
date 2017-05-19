'''
Created on 2 avr. 2017

@author: Naitra
'''

import matplotlib.pyplot as plt
from numpy import exp
from numpy import transpose, sum, array, linspace, random, maximum

from CreditModel.DirectionalWayRisk.Weights import Merton, Probabilities_CVA
from CreditModel.Tools.RiskStatistics import risk_statistics
from Scripts.portfolio import create_contracts
from StochasticProcess.Commodities.Schwartz97 import Schwartz97

if __name__ == '__main__':
    pass

###################
### Parameters
###################

# parameters schwartz
S0 = 45
delta0 = 0.15
r = 0.1
sigma_s = 0.393
kappa = 1.876
sigma_e = 0.527
corr = 0.766
lamb = 0.198
alpha = 0.106 - lamb / kappa
model = Schwartz97(r=r, sigma_s=sigma_s, kappa=kappa, alpha_tilde=alpha, sigma_e=sigma_e, rho=corr)

# simulation parameter

# Simulation
Nsimulations = 10
random.seed(128)

nb_point_exposure = 24
nb_point_path = 20
nb_point_intensity = 5
start_exposure = 0.05
start_default = 0.1
start_path = 0.01

book, name_contracts, time_exposure1 = create_contracts(mdl=model, S_ini=S0)
book = book  # [20:30]

T_horizon_CVA = max(time_exposure1)

time_exposure2 = linspace(start_exposure, T_horizon_CVA, nb_point_exposure,
                          endpoint=True)  # times at which we want exposure
time_intensity = linspace(start_exposure, T_horizon_CVA, nb_point_intensity,
                          endpoint=True)  # times where calibrated intensity breaks
time_path = linspace(start_path, T_horizon_CVA, nb_point_path, endpoint=True)

timeline = sorted(set(time_exposure1) | set(time_exposure2) | set(time_path) | set(time_intensity))
time_exposure = sorted(set(time_exposure1) | set(time_exposure2))
index_exposure = [timeline.index(t) for t in time_exposure]

Ntimes = len(timeline)
range_exposure = range(0, len(time_exposure))

###################
### Compute probabilities
###################

constant_intensity = 0.001  # have to be calibrated from cds
hazard_rates = array([constant_intensity for k in timeline])
PD = Probabilities_CVA(hazard_rates=hazard_rates, timeline=array(timeline), times_exposure=time_exposure)
# probability_default_value = exp(-constant_intensity*time_intensity)

# PD = [piecewise_flat(t,probability_default_value,time_intensity) for t in time_exposure]

###################
### Simulation
###################


# market sensitivities

simulated_paths = model.PathQ(S_ini=S0, delta_ini=delta0, timeline=timeline, nb_path=Nsimulations)
simulated_paths = transpose(simulated_paths, (2, 1, 0))
S = simulated_paths[0]
convenience_yield = simulated_paths[1]

###################
### Portfolio
###################

# portfolio
# cumulated_price = lambda s, t: sum([instrument(s, t) for instrument in portfolio], axis=0)
nb_instrument = len(book)
range_portfolio = range(0, nb_instrument)
V = [[instrument(timeline[t], S[t, :], convenience_yield[t, :]) for t in index_exposure] for instrument in book]
# Vtot = [cumulated_price(simulated_paths[t,:], timeline[t]) for t in index_exposure]
###################
### Exposure
###################

Collateral_level = 0
Exposure_function = lambda V: maximum(V - Collateral_level, 0)
Exposures = [[Exposure_function(V[k][t]) for t in range_exposure] for k in range_portfolio]
# Exposurestot = [Exposure_function(Vtot[t]) for t in range_exposure]

###################
### Compute weights
###################
rho = 0.5
dfs = [Merton(Exposures[1][t], rho, PD[t], tolerance=0.001) for t in range_exposure]
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
# resultsDWRtot = array([risk_statistics(Exposures[t], weights=weightsMerton[t], alpha=alpha) for t in range_exposure])
# first dimension is time

###################
### cva
###################

DiscountFactorXdefault = exp(-r * array(time_exposure)) * PD
CVA_Indep = [sum(resultsIndep[k, :, 0] * DiscountFactorXdefault) for k in range_portfolio]
CVA_DWR = [sum(resultsDWR[k, :, 0] * DiscountFactorXdefault) for k in range_portfolio]

###################
### plots
###################

fig = plt.figure()

risk_measure = ['Expected Exposure', 'PFE' + str(confidence), 'PFE' + str(1 - confidence), 'ES' + str(confidence)]
for k in range(len(risk_measure)):  # 3 is the number of parameters
    ax = plt.subplot(len(risk_measure) / 2, 2, k + 1)
    plt.plot(time_exposure, resultsIndep[0, :, k], 'blue', label='Indep')
    plt.plot(time_exposure, resultsDWR[0, :, k], 'red', label='DRW')
    plt.title(risk_measure[k])
    plt.xlabel('dates')
    plt.legend()
    # plt.xticks(x, tenors[t,:],rotation='vertical')
# plt.figlegend( fig, ['Indep', 'WWR'], 'upper right' )

# plt.title("Risk measure against time")
# fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=3, fancybox=True, shadow=True)
# fig.legend(['Indep','DWR'])
plt.show()

# plt.plot(time_exposure, array(Exposures), color='green', marker='o', markerfacecolor='None', linestyle ='None')
