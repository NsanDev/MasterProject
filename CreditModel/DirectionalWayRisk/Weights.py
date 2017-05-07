from bisect import bisect_left

from numpy import sum, sqrt, vectorize, dot, exp, zeros, mean, log
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF

'''
Compute weights from the Merton Model
'''


def Merton(Z_M, rho, probability_default, tolerance=0.001):

    C = norm.ppf(probability_default)
    # market factor
    ecdf_Z_M = ECDF(Z_M)
    def set_inf_to_mean(x):
        if x == float('Inf'):
            return 1 - tolerance
        elif x == -float('Inf'):
            return tolerance
        else:
            return x
    set_inf_to_mean = vectorize(set_inf_to_mean)
    Z = set_inf_to_mean(norm.ppf(ecdf_Z_M(Z_M)))

    denom = sqrt(1 - rho * rho)
    weights = norm.cdf((C - rho * Z) / denom)
    weights = weights / sum(weights)

    return weights


'''
Integrate intensity from 0 to t. intensity is defined as a piecewise function times,h_rates
from 0 to times[0] h_rates return h[0]
from times[k-1] to times[k] h_rates return h[k] 
the implementation is such that: assert (t<=times[-1])
'''
def _integrate_intensity(t, h_rates, timeline):

    assert (len(timeline) == h_rates.shape[0])
    assert (timeline[0] > 0)  # assume zero is not in the list of time where we want to calculate exposure
    assert (t <= timeline[-1])  # last element should be higher than t

    max_index = bisect_left(timeline, t)
    if max_index == 0:
        return t * h_rates[0, ]
    else:
        result = h_rates[0,] * timeline[0]
        if max_index > 1:
            result = result + dot(timeline[1:max_index] - timeline[0:max_index - 1], h_rates[1:max_index, ])
        result = result + (t - timeline[max_index - 1]) * h_rates[max_index,]
        return result

'''
Compute weights from simulated hazard rate. (for Hull and Ruiz approach)
'''


def Weights(hazard_rates, timeline, times_exposure):

    assert (timeline[0]>0)
    assert (all(t in timeline for t in times_exposure))
    assert (all(timeline[i] < timeline[i + 1] for i in range(0, len(timeline) - 1)))
    assert (hazard_rates.shape[0] == len(timeline))

    result = [-_integrate_intensity(t, hazard_rates, timeline) for t in times_exposure]
    result = exp(result)
    for t in range(len(timeline),0,-1):
        result[t] = result[t-1] - result[t]
        result[t] = result[t]/sum(result[t])  # normalization
    result[0] = result[0]/sum(result[0])
    #TODO check that weights are smaller than 1 and add up to 1
    return result

'''
This calibrator can be reused for specific + global way risk model
b: defined in Hull 2012
Z: Market factor or transformation of Market factor (more generic than value of portfolio)
probability_default: P(tau<t) for t in times_default
times_default: times to define P(tau<t) as a piecewise flat function
'''


def Calibration_hull(b, Z, timeline, probability_default, times_default):

    assert (timeline[0]>0)
    assert (all(t in timeline for t in times_default))
    assert (all(timeline[i] < timeline[i + 1] for i in range(0, len(timeline) - 1)))
    assert (Z.shape[0] == len(timeline))

    a = zeros(len(times_default))
    a[0] = -log(mean(exp(-_integrate_intensity(times_default[0], Z,timeline)))/
               probability_default[0])/times_default[0]
    cumulative_a = times_default[0]*a[0]
    for k in range(1, len(times_default)):
        a[k] = -log(mean(exp(- cumulative_a - _integrate_intensity(times_default[k], Z, timeline)))/probability_default[k]) /(times_default[k] - times_default[k - 1])
        cumulative_a = cumulative_a + a[k] * (times_default[k] - times_default[k - 1])
    # TODO test that
    return a
