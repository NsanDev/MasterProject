from numpy import array, linspace

from CreditModel.DirectionalWayRisk.Weights import Hull
from Scripts.data_generators.parameters import Q_default, discount_factor, save_array, load_array, cumulated_Q_default

###################
### Load Data saved in folder data
###################
timeline = load_array('timeline')
time_exposure = load_array('time_exposure')
S = load_array('spot_prices')
convenience_yield = load_array('convenience_yields')
V_alltimes = load_array('contracts_alltimes')
Exposures_alltimes = load_array('exposures_alltimes')
index_exposure = [list(timeline).index(t) for t in time_exposure]
###################
### Choose Parameters on Hull
###################
Z_M = V_alltimes  # need all times, not only at exposure

DiscountFactorXdefault = Q_default(time_exposure) * discount_factor(time_exposure)
range_exposure = range(0, len(time_exposure))
range_portfolio = range(0, len(V_alltimes))
PD = cumulated_Q_default(time_exposure)

###################
### Compute cva
###################
bs = linspace(start=-0.001, stop=0.001, num=11, endpoint=True)


def calc_cva_hull(b, Z_M=Z_M):
    weightsHull = [Hull(b, Z=Z_M[k], timeline=timeline, probability_default=PD, times_default=time_exposure,
                        times_exposure=time_exposure) for k in range_portfolio]
    resultsDWR = array(
        [[sum(Z_M[k][index_exposure[t]] * weightsHull[k][t]) for t in range_exposure] for k in range_portfolio])
    return [sum(resultsDWR[k, :] * DiscountFactorXdefault) for k in range_portfolio]


cva_hull = [calc_cva_hull(b, Z_M=Z_M) for b in bs]
save_array('bs_hull', bs)
save_array('cva_hull', cva_hull)

# TODO: finish that cva calculator for different values of rho (Loop).
# TODO: Think about the cases where rho = 0 or rho = 1
