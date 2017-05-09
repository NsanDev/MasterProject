from numpy import array, floor

from StochasticProcess.Commodities.Schwartz97 import *

'''
Create portfolio with different contracts:
- forward
- european call on forward
- european put om forward
- swap with exchanged time t_i every M month and delivery M months+t_i with t_i max = T_horizon 
return list of functions of t,s,delta
'''


def create_contracts(mdl: Schwartz97, S_ini):
    # T_M for forward: every month until T_horizon
    T_horizon = 1
    # maturities of option: every month from T_maturity_start until T_horizon
    T_maturity_start = 1 / 2
    # foreach maturity of option, forward with delivery date starting from 1 month later until
    # nb_month_max_option later(included)
    nb_month_max_option = 6
    # Strike +- 6 relative to S0
    strike_option = array([k for k in range(-6, 6 + 1)]) + S_ini

    book = []
    names_contracts = []

    #####################
    ### forward T_M
    #####################


    time_forward = array([k / 12 for k in range(1, int(
        floor(12 * T_horizon)) + 1)])  # linspace(T_horizonCVA, T_horizonCVA+1, endpoint=True)
    relevant_times = set(time_forward)
    for T_M in time_forward:
        book.append(lambda t, s, delta: mdl.forward(t=t, T=T_M, S_ini=s, delta_ini=delta))
        names_contracts.append("Forward $T_M$=" + str(T_M))

    #####################
    ### call put
    #####################

    maturity_option = time_forward
    relevant_times = relevant_times | set(maturity_option)
    time_forward_option = time_forward + T_maturity_start
    for T in maturity_option:
        for K in strike_option:
            for k in range(1, nb_month_max_option + 1):
                book.append(lambda t, s, delta: mdl.call(t, maturity_option=T, delivery_time_forward=T + k / 12, \
                                                         K=K, S_ini=s, delta_ini=delta))
                book.append(lambda t, s, delta: mdl.put(t, maturity_option=T, delivery_time_forward=T + k / 12, \
                                                        K=K, S_ini=s, delta_ini=delta))
                names_contracts.append("Call T=" + str(maturity_option) + ", K="
                                       + str(K) + ", T_M=T+" + str(k) + "Months")
                names_contracts.append("Put T=" + str(maturity_option) + ", K="
                                       + str(K) + ", T_M=T+" + str(k) + "Months")

    #####################
    ### Swap
    #####################

    # monthly swap
    exchange_time = array([k / 12 for k in range(1, int(floor(12 * T_horizon)) + 1)])
    relevant_times = relevant_times | set(exchange_time)
    time_delivery = exchange_time + (12 / 1) ** -1
    book.append(lambda t, s, delta: mdl.swap(t=t, exchange_time=exchange_time,
                                             maturities=time_delivery, S_ini=s, delta_ini=delta))
    names_contracts.append("Monthly Swap T=" + str(maturity_option))

    # 3-months swap
    exchange_time = array([k / 4 for k in range(1, int(floor(12 / 4 * T_horizon)) + 1)])
    relevant_times = relevant_times | set(exchange_time)
    time_delivery = exchange_time + (12 / 4) ** -1
    book.append(lambda t, s, delta: mdl.swap(t=t, exchange_time=exchange_time,
                                             maturities=time_delivery, S_ini=s, delta_ini=delta))
    names_contracts.append("3-months Swap T=" + str(maturity_option))

    # 4-months swap
    exchange_time = array([k / 3 for k in range(1, int(floor(12 / 3 * T_horizon)) + 1)])
    relevant_times = relevant_times | set(exchange_time)
    time_delivery = exchange_time + (12 / 3) ** -1
    book.append(lambda t, s, delta: mdl.swap(t=t, exchange_time=exchange_time,
                                             maturities=time_delivery, S_ini=s, delta_ini=delta))
    names_contracts.append("4-months Swap T=" + str(maturity_option))

    # 6-months swap
    exchange_time = array([k / 6 for k in range(1, int(floor(12 / 6 * T_horizon)) + 1)])
    relevant_times = relevant_times | set(exchange_time)
    time_delivery = exchange_time + (12 / 6) ** -1
    book.append(lambda t, s, delta: mdl.swap(t=t, exchange_time=exchange_time,
                                             maturities=time_delivery, S_ini=s, delta_ini=delta))
    names_contracts.append("6-months Swap T=" + str(maturity_option))

    # book = [vectorize(instr) for instr in portfolio]
    # cumulated_price = lambda s, t: sum([instrument(s, t) for instrument in portfolio], axis=0)


    relevant_times = sorted(relevant_times)
    return (book, names_contracts, relevant_times)
