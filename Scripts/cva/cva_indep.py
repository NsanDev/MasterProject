from numpy import mean

from Scripts.parameters import Q_default, discount_factor, save_array, load_array

###################
### Load Data saved in folder data
###################
timeline = load_array('timeline')
time_exposure = load_array('time_exposure')
S = load_array('spot_prices')
convenience_yield = load_array('convenience_yields')
V = load_array('contracts')
Exposures = load_array('exposures')

###################
### Choose Parameters on merton
###################
Z_M = Exposures

DiscountFactorXdefault = Q_default(time_exposure) * discount_factor(time_exposure)
range_exposure = range(0, len(time_exposure))
range_portfolio = range(0, len(V))

###################
### Compute cva
###################
resultsDWR = mean(Z_M, axis=2)
cva_indep = [sum(resultsDWR[k, :] * DiscountFactorXdefault) for k in range_portfolio]
save_array('cva_indep', cva_indep)
