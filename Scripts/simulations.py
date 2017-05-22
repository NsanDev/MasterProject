import time

from numpy import array, abs

from Scripts.parameters import load_model, simulate_path, portfolio, save_array, exposure_function, S0, delta0, shift_S


def launch_simulation(S_ini=S0, delta_ini=delta0, shift_str=''):
    ###################
    ### Initialize
    ###################

    model = load_model()
    book, time_exposure, timeline, contract_names = portfolio()
    index_exposure = [list(timeline).index(t) for t in time_exposure]
    range_exposure = range(0, len(time_exposure))

    range_book = range(0, len(contract_names))
    range_timeline = range(0, len(timeline))

    ###################
    ### Simulation
    ###################

    # market sensitivities
    simulated_paths = simulate_path(timeline, S_ini=S_ini, delta_ini=delta_ini, reset_seed=True)
    S = simulated_paths[0]

    convenience_yield = simulated_paths[1]

    ###################
    ### Portfolio
    ###################

    contracts_alltimes = array([[instrument(timeline[t], S[t, :], convenience_yield[t, :]) for t in range_timeline]
                                for instrument in book])
    contract_ini = array([instrument(0, S_ini, delta_ini) for instrument in book])
    contracts = contracts_alltimes[:, index_exposure, :]

    ###################
    ### Exposure
    ###################

    exposure_alltimes = exposure_function(contracts_alltimes)
    exposures = exposure_alltimes[:, index_exposure, :]

    ###################
    ### Market Factor for hull()
    ###################

    Z_S = S / S_ini
    Z_V = array([contracts_alltimes[k] / abs(contract_ini[k]) for k in range_book])
    Z_E = array([exposure_alltimes[k] / abs(contract_ini[k]) for k in range_book])

    save_array('spot_prices' + shift_str, S)
    save_array('convenience_yields' + shift_str, convenience_yield)
    save_array('contracts' + shift_str, contracts)
    save_array('exposures' + shift_str, exposures)
    # save_array('contracts_alltimes'+shift_str, contracts_alltimes)
    # save_array('exposures_alltimes'+shift_str, exposure_alltimes)
    save_array('Z_S' + shift_str, Z_S)
    save_array('Z_V' + shift_str, Z_V)
    save_array('Z_E' + shift_str, Z_E)

    save_array('timeline', timeline)
    save_array('time_exposure', time_exposure)
    save_array('contract_names', contract_names)


start_time = time.clock()
launch_simulation()
launch_simulation(S_ini=S0 + shift_S, shift_str='_shift_S_pos')
launch_simulation(S_ini=S0 - shift_S, shift_str='_shift_S_neg')
time_perf = time.clock() - start_time
print(time_perf)
