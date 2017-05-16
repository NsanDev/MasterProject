import time

from numpy import array

from Scripts.parameters import load_model, simulate_path, portfolio, save_array, exposure_function

start_time = time.clock()
###################
### Initialize
###################

model = load_model()
book, time_exposure, timeline, contract_names = portfolio()
book = book  # [20:30]
index_exposure = [list(timeline).index(t) for t in time_exposure]
range_exposure = range(0, len(time_exposure))

range_book = range(0, len(contract_names))
range_timeline = range(0, len(timeline))

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
contracts_alltimes = array([[instrument(timeline[t], S[t, :], convenience_yield[t, :]) for t in range_timeline]
                            for instrument in book])
contracts = contracts_alltimes[:, index_exposure, :]
###################
### Exposure
###################
exposure_alltimes = exposure_function(contracts_alltimes)
exposures = exposure_alltimes[:, index_exposure, :]

save_array('timeline', timeline)
save_array('time_exposure', time_exposure)
save_array('spot_prices', S)
save_array('convenience_yields', convenience_yield)
save_array('contract_names', contract_names)
save_array('contracts', contracts)
save_array('exposures', exposures)
save_array('contracts_alltimes', contracts_alltimes)
save_array('exposures_alltimes', exposure_alltimes)

timeperf = time.clock() - start_time
print(timeperf)
