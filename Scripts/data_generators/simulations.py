import time

from Scripts.data_generators.parameters import load_model, simulate_path, portfolio, save_array, exposure_function

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
Exposures = [[exposure_function(contract[t]) for t in range_exposure] for contract in V]

save_array('timeline', timeline)
save_array('time_exposure', time_exposure)
save_array('spot_prices', S)
save_array('convenience_yields', convenience_yield)
save_array('contracts', V)
save_array('contract_names', contract_names)
save_array('exposures', Exposures)

timeperf = time.clock() - start_time
print(timeperf)
