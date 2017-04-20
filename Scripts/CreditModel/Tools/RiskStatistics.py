from math import floor
from numpy import array, ones, dot, sum


def risk_statistics(values, weights=None, alpha=0.05):

    nb_values = len(values)
    weights = ones(nb_values) / nb_values if weights is None else weights

    # array of [value,weight] sorted by value
    sorted_vw = array(sorted([[values[k], weights[k]] for k in range(0, nb_values)], key=lambda x: x[0]))

    index_alpha = 0
    cumulative_weights = 0
    while index_alpha < nb_values and cumulative_weights < 1.0 - alpha:
        cumulative_weights = cumulative_weights + sorted_vw[index_alpha, 1]
        index_alpha = index_alpha + 1

    if index_alpha > 0: index_alpha = index_alpha - 1

    value_at_risk = sorted_vw[index_alpha, 0]
    expected_shortfall = dot(sorted_vw[index_alpha:, 0], sorted_vw[index_alpha:, 1]) / sum(sorted_vw[index_alpha:,1])
    expected_exposure = sum(values*weights)

    return [expected_exposure, value_at_risk, expected_shortfall ]