import time

from numpy import maximum, save

from Scripts.CVA.parameters import load_model, simulate_path, portfolio

start_time = time.clock()
###################
### Initialize
###################

model = load_model()
book, time_exposure, timeline, contract_names = portfolio()
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
V = [[instrument(timeline[t], S[t, :], convenience_yield[t, :]) for t in index_exposure] for instrument in book]

###################
### Exposure
###################
Collateral_level = 0
Exposure_function = lambda V: maximum(V - Collateral_level, 0)
Exposures = [[Exposure_function(contract[t]) for t in range_exposure] for contract in V]

save('timeline', timeline)
save('time_exposure', time_exposure)
save('spot_prices', S)
save('convenience_yields', convenience_yield)
save('contracts', V)
save('contract_names', contract_names)
save('exposures', Exposures)

timeperf = time.clock() - start_time
print(timeperf)
