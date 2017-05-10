from numpy import maximum

from Scripts.CVA.parameters import load_model, simulate_path, portfolio

###################
### Initialize
###################

model = load_model()
book, time_exposure, timeline, contract_name = portfolio()
nb_contracts = len(contract_name)
book = book  # [20:30]
index_exposure = [list(timeline).index(t) for t in time_exposure]
range_exposure = range(0, len(time_exposure))

###################
### Simulation
###################
# market sensitivities
simulated_paths = simulate_path(timeline)
S = simulated_paths[0]
convenience_yield = simulated_paths[1]

###################
### Portfolio
###################

# cumulated_price = lambda s, t: sum([instrument(s, t) for instrument in portfolio], axis=0)

range_portfolio = range(0, nb_contracts)
V = [[instrument(timeline[t], S[t, :], convenience_yield[t, :]) for t in index_exposure] for instrument in book]

###################
### Exposure
###################
Collateral_level = 0
Exposure_function = lambda V: maximum(V - Collateral_level, 0)
Exposures = [[Exposure_function(V[k][t]) for t in range_exposure] for k in range_portfolio]
