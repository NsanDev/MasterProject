from numpy import array, ones, dot
import numpy as np

def risk_statistics(values, weights=None, alpha=0.05):

    nb_values = len(values)
    weights = ones(nb_values) / nb_values if weights is None else weights

    # array of [value,weight] sorted by value
    sorted_vw = array(sorted([[values[k], weights[k]] for k in range(0, nb_values)], key=lambda x: x[0]))

    index_alpha_ini = 0
    index_alpha_end = nb_values-1
    cumulative_weights_ini = 0
    cumulative_weights_end = 1
    while index_alpha_ini < nb_values and cumulative_weights_ini + np.finfo(float).eps < alpha:
        cumulative_weights_ini = cumulative_weights_ini + sorted_vw[index_alpha_ini, 1]
        index_alpha_ini = index_alpha_ini + 1

    while index_alpha_end > 0 and cumulative_weights_end >= 1.0 - alpha:
        cumulative_weights_end = cumulative_weights_end - sorted_vw[index_alpha_end, 1]
        index_alpha_end = index_alpha_end - 1

    if index_alpha_ini > 0: index_alpha_ini = index_alpha_ini - 1

    expected_exposure = np.sum(values * weights)
    value_at_risk_ini = sorted_vw[index_alpha_ini, 0]
    value_at_risk_end = sorted_vw[index_alpha_end, 0]
    expected_shortfall = dot(sorted_vw[index_alpha_end:, 0] , sorted_vw[index_alpha_end:, 1]) / np.sum(sorted_vw[index_alpha_end:,1])

    return [expected_exposure, value_at_risk_ini, value_at_risk_end, expected_shortfall]
