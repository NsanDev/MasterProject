import matplotlib.pyplot as plt
from numpy import sum, array, linspace, random, floor, ones
from CreditModel.Tools.RiskStatistics import risk_statistics
from CreditModel.DirectionalWayRisk.Weights import Hull
from Scripts.parameters import model, simulate_path, exposure_function, S0, delta0, cumulated_Q_survival, Q_default, \
    discount_factor
from numpy import histogram
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from time import clock

if __name__ == '__main__':
    pass

    ###################
    ### Parameters
    ###################
    Nsimulations = 100000
    random.seed(128)
    nb_point_exposure = 20
    start_path = 0.01
    T_horizon = 1
    b_S = [-0.5, -0.5, 0, 0.5]
    b_V = [0, 0.5, 0.5, 0.5]
    time_obs_exp = 0.75

    #####################
    ### Create books
    #####################

    book = []
    names_contracts = []
    K = 0.9 * S0
    T_delivery = T_horizon + 1 / 12
    book.append(lambda t, s, delta, T_M=T_horizon: model.forward(t=t, T=T_M, S_ini=s, delta_ini=delta))
    book.append(lambda t, s, delta: model.call(t, maturity_option=T_horizon, delivery_time_forward=T_delivery,
                                               K=K, S_ini=s, delta_ini=delta))
    book.append(lambda t, s, delta: model.put(t, maturity_option=T_horizon, delivery_time_forward=T_delivery,
                                              K=K, S_ini=s, delta_ini=delta))
    # 3-months swap
    exchange_time = array([k / 4 for k in range(1, int(floor(4 * T_horizon)) + 1)])
    time_delivery = exchange_time + (12 / 4) ** -1
    book.append(lambda t, s, delta, exchange_time=exchange_time, time_delivery=time_delivery
                : model.swap(t=t, exchange_time=exchange_time, maturities=time_delivery,
                             fixed_legs=K * ones(len(exchange_time)), S_ini=s, delta_ini=delta))
    names_contracts.append("Quart. Swap T=" + str(T_horizon) + " K=" + str(K))
    names_contracts.append("Future $T_M$=" + str(T_horizon))
    names_contracts.append("Call T=" + str(T_horizon) + ", K=" + str(K) + ", T_M=" + str(T_delivery))
    names_contracts.append("Put T=" + str(T_horizon) + ", K=" + str(K) + ", T_M=" + str(T_delivery))
    range_book = range(0, len(book))
    #####################
    ### timeline
    #####################

    time_exposure = linspace(start_path, T_horizon, nb_point_exposure, endpoint=True)  # times at which we want exposure
    time_exposure = sorted(set(exchange_time) | set(time_exposure) | set([time_obs_exp]))
    timeline = time_exposure  # Special case
    index_exposure = [timeline.index(t) for t in time_exposure]
    index_obs_exp = time_exposure.index(time_obs_exp)
    time_exposure = array(time_exposure)
    timeline = array(timeline)
    Ntimes = len(timeline)
    range_timeline = range(0, Ntimes)
    range_exposure = range(0, len(time_exposure))

    #####################
    ### MC simulation
    #####################

    simulated_paths = simulate_path(timeline, S_ini=S0, delta_ini=delta0, reset_seed=True, nb_path=Nsimulations)
    S = simulated_paths[0]
    convenience_yield = simulated_paths[1]
    contracts_alltimes = array([[instrument(timeline[t], S[t, :], convenience_yield[t, :]) for t in range_timeline]
                                for instrument in book])
    contract_ini = array([instrument(0, S0, delta0) for instrument in book])
    contracts = contracts_alltimes[:, index_exposure, :]

    ###################
    ### Exposure
    ###################
    exposure_alltimes = exposure_function(contracts_alltimes)
    exposures = exposure_alltimes[:, index_exposure, :]

    ###################
    ### Market Factor for hull()
    ###################

    Z_S = S / S0
    Z_V = array([contracts_alltimes[k] / abs(contract_ini[k]) for k in range_book])

    from joblib import Parallel, delayed
    from multiprocessing import cpu_count

    time_default = array([T_horizon])
    survival_probability = cumulated_Q_survival(time_default)


    def weights_Hull(bs, bv):
        Z_M = bs * Z_S + bv * Z_V
        return Parallel(n_jobs=cpu_count(), backend="threading")(
            delayed(Hull)(Z=Z_M[l], timeline=timeline, survival_probability=survival_probability,
                          times_survival=time_default, times_exposure=time_exposure)
            for l in range(0, len(Z_M)))


    def create_fig(bs, bv):
        # fig = plt.figure()
        weights = weights_Hull(bs, bv)
        ###################
        ### Compute stats
        ###################

        confidence = 0.05
        resultsIndep = array(
            [[risk_statistics(exposures[k][t], alpha=confidence) for t in range_exposure] for k in range_book])

        resultsDWR = array(
            [[risk_statistics(exposures[k][t], weights=weights[k][t], alpha=confidence) for t in range_exposure] for k
             in
             range_book])

        DiscountFactorXdefault = Q_default(time_exposure) * discount_factor(time_exposure)
        CVA_Indep = [sum(resultsIndep[k, :, 0] * DiscountFactorXdefault) for k in range_book]
        CVA_DWR = [sum(resultsDWR[k, :, 0] * DiscountFactorXdefault) for k in range_book]

        ###################
        ### plots
        ###################

        # fig = plt.figure()

        risk_measure = ['Expected Exposure', 'PFE' + str(confidence), 'PFE' + str(1 - confidence),
                        'ES' + str(confidence)]
        linestyles = ['-', '-.', '-.', ':']
        f, axes = plt.subplots(4, 3, figsize=(12, 15))
        for j in range_book:  # id of the contract
            for k in range(len(risk_measure) - 1):  # 3 is the number of parameters
                axes[j, 0].plot(time_exposure, resultsIndep[j, :, k], color='blue', ls=linestyles[k], linewidth=2)
                axes[j, 0].plot(time_exposure, resultsDWR[j, :, k], color='red', ls=linestyles[k], linewidth=2)
                if j != len(range_book) - 1:
                    axes[j, 0].tick_params(labelbottom='off')

            axes[j, 1].plot(time_exposure, resultsIndep[j, :, 3], color='blue', ls=linestyles[3], linewidth=2)
            axes[j, 1].plot(time_exposure, resultsDWR[j, :, 3], color='red', ls=linestyles[3], linewidth=2)

            val = array(exposures[j][index_obs_exp])
            hist1, bin1 = histogram(val, bins='auto', density=True)
            axes[j, 2].hist(val, bins=bin1, histtype='step', normed=True)
            axes[j, 2].hist(val, bins=bin1, histtype='step', weights=weights[j][index_obs_exp], color='red',
                            normed=True)
            axes[j, 2].ticklabel_format(axis='y', style='sci', scilimits=(0, 1))
        if j == len(range_book) - 1:
            axes[j, 0].set_xlabel('time')
            axes[j, 1].set_xlabel('time')
            axes[j, 2].set_xlabel('prices')
        contract_name_short = ["Future", "Call", "Put", "Swap"]
        for j in range_book:
            axes[j, 0].set_ylabel(contract_name_short[j], fontweight='bold', fontsize=12)

        red_patch = mpatches.Patch(color='red', label='DWR')
        blue_patch = mpatches.Patch(color='blue', label='Indep.')
        ee_patch = mlines.Line2D([], [], color='black', label='', ls='-')
        pfe_patch = mlines.Line2D([], [], color='black', label=r'PFE_0.5', ls='-.')
        es_patch = mlines.Line2D([], [], color='black', label=r'ES_0.5', ls=':')

        f.legend(handles=[blue_patch, red_patch, ee_patch, pfe_patch, es_patch]
                 , labels=['Indep.', 'DWR', 'EE', r'$PFE_{0.5}$', r'$ES_{0.5}$']
                 , ncol=5, loc='lower center', fontsize=14)
        title = r"$b_S$ = {} , $b_V$ = {}".format(str(bs), str(bv))

        f.suptitle(title, fontweight='bold', fontsize=16)
        # plt.savefig(title+".png")
        return f


    start = clock()
    result = [create_fig(b_S[k], b_V[k]) for k in range(len(b_S))]
    for k in range(len(result)):
        result[k].savefig(f"pictures/{k}.png")

    total_time = clock() - start
    print(total_time)
