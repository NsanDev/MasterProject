from numpy import array

from CreditModel.DirectionalWayRisk.Weights import Merton
from Scripts.data_generators.parameters import Q_default, discount_factor, save_array, load_array

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
PD = Q_default(time_exposure)


###################
### Compute cva
###################

def calc_cva_merton(rho, Z_M=Z_M):
    weightsMerton = [[Merton(Z_M[k][t], rho, PD[t], tolerance=0.001) for t in range_exposure] for k in range_portfolio]
    resultsDWR = array([[sum(Z_M[k][t] * weightsMerton[k][t]) for t in range_exposure] for k in range_portfolio])
    return [sum(resultsDWR[k, :] * DiscountFactorXdefault) for k in range_portfolio]


rhos = [k / 10 for k in range(-9, 10)]
cva_merton = [calc_cva_merton(rho, Z_M=Z_M) for rho in rhos]
save_array('rhos_merton', rhos)
save_array('cva_merton', cva_merton)

# TODO: finish that cva calculator for different values of rho (Loop).
    # TODO: Think about the cases where rho = 0 or rho = 1