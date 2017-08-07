from numpy import array, floor, ones

from StochasticProcess.Commodities.Schwartz97 import *

'''
Create portfolio with different contracts:
- forward
- european call on forward
- european put om forward
- swap with exchanged time t_i every M month and delivery M months+t_i with t_i max = T_horizon 
return list of functions of t,s,delta
'''


def create_contracts(mdl: Schwartz97, S0):
    # T_M for forward: every month until T_horizon
    T_horizon = 1
    # maturities of option: every month from T_maturity_start until T_horizon
    T_maturity_start = 1 / 2
    # foreach maturity of option, forward with delivery date starting from 1 month later until
    # nb_month_max_option later(included)
    nb_month_max_option = 4
    # Strike +- 6 relative to S0
    strike_option = array([2 * k for k in range(-3, 3 + 1)]) + S0
    # maturity of options
    maturity_option = [0.25, 0.5, 1]
    # delivery time after exercise (have to be indicated in number of month)
    delivery_after_exercise = [1, 3]

    book = []
    names_contracts = []

    #####################
    ### forward T_M
    #####################


    time_forward = array([k / 12 for k in range(1, int(
        floor(12 * T_horizon)) + 1)])  # linspace(T_horizonCVA, T_horizonCVA+1, endpoint=True)
    cashflow_times = set(time_forward)
    for T_M in time_forward:
        book.append(lambda t, s, delta, T_M=T_M: mdl.forward(t=t, T=T_M, S_ini=s, delta_ini=delta))
        names_contracts.append("Forward $T_d$=" + str(T_M))

    #####################
    ### call put
    #####################


    cashflow_times = cashflow_times | set(maturity_option)
    time_forward_option = time_forward + T_maturity_start
    for T in maturity_option:
        for K in strike_option:
            for k in delivery_after_exercise:
                book.append(lambda t, s, delta, T=T, T_M=T_M, K=K, k=k:
                            mdl.call(t, maturity_option=T, delivery_time_forward=T + k / 12,
                                     K=K, S_ini=s, delta_ini=delta))
                book.append(lambda t, s, delta, T=T, T_M=T_M, K=K, k=k:
                            mdl.put(t, maturity_option=T, delivery_time_forward=T + k / 12,
                                    K=K, S_ini=s, delta_ini=delta))
                names_contracts.append("Call T=" + str(T) + ", K="
                                       + str(K) + ", $T_d$=T+" + str(k) + "Months")
                names_contracts.append("Put T=" + str(T) + ", K="
                                       + str(K) + ", $T_d$=T+" + str(k) + "Months")

    #####################
    ### Swap
    #####################
    def append_swap(periodicity, cashflow_times, months_later, fixed_leg):
        exchange_time = array([k / (12 / periodicity) for k in range(1, int(floor(12 / periodicity * T_horizon)) + 1)])
        cft = cashflow_times | set(exchange_time)
        time_delivery = exchange_time + months_later  # delivery in one month after exchange leg
        book.append(lambda t, s, delta, exchange_time=exchange_time, time_delivery=time_delivery
                    : mdl.swap(t=t, exchange_time=exchange_time, maturities=time_delivery,
                               fixed_legs=fixed_leg * ones(len(exchange_time)), S_ini=s, delta_ini=delta))
        names_contracts.append(
            "Swap" + " K=" + str(fixed_leg / S0) + "$S_0$" + " Periodicity=" + str(periodicity) + " Months", )
        return cft

    periodicities = [3, 4, 6, 12]
    months_delivery_after_exchange = [1, 3, 4]
    fixed_leg = array([0.8, 0.82, 0.84, 0.86, 0.88, 0.9]) * S0

    for p in periodicities:
        for m in months_delivery_after_exchange:
            for K in fixed_leg:
                cashflow_times = append_swap(p, cashflow_times, m, K)



    cashflow_times = sorted(cashflow_times)
    return (book, cashflow_times, names_contracts)
