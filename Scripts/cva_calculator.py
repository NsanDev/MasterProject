from numba import jit
from numpy import mean, zeros, empty, array

from CreditModel.DirectionalWayRisk.Weights import Merton, Hull
from Scripts.parameters import discount_factor, save_array, load_array

###################
### Load Data saved in folder data
###################
Exposures = load_array('exposures')
timeline = load_array('timeline')
time_exposure = load_array('time_exposure')
time_default = [timeline[-1]]
range_exposure = range(0, len(time_exposure))
range_portfolio = range(0, len(Exposures))
index_exposure = [list(timeline).index(t) for t in time_exposure]


def generate_cva_indep(Q_default, name_saved_file):
    PD = Q_default(time_exposure)
    DiscountFactorXdefault = PD * discount_factor(time_exposure)

    ###################
    ### Compute cva
    ###################
    resultsDWR = mean(Exposures, axis=2)
    cva_indep = [sum(resultsDWR[k, :] * DiscountFactorXdefault) for k in range_portfolio]
    save_array(name_saved_file, cva_indep)


def generate_cva_merton(Z_M, rhos_merton, Q_default, name_saved_file):
    PD = Q_default(time_exposure)
    DiscountFactorXdefault = PD * discount_factor(time_exposure)

    @jit
    def calc_cva_merton(rho):
        result = zeros(Z_M.shape[0])
        for k in range_portfolio:
            for t in range_exposure:
                result[k] = result[k] + DiscountFactorXdefault[t] * \
                                        sum(Z_M[k][t] * Merton(Z_M[k][t], rho, PD[t], tolerance=0.001))
        return result

    # it is faster with jit (but just a little bit)
    # def calc_cva_merton0(rho, Z_M=Z_M):
    #    weightsMerton = [[Merton(Z_M[k][t], rho, PD[t], tolerance=0.001) for t in range_exposure] for k in range_portfolio]
    #    resultsDWR = array([[sum(Z_M[k][t] * weightsMerton[k][t]) for t in range_exposure] for k in range_portfolio])
    #    return [sum(resultsDWR[k, :] * DiscountFactorXdefault) for k in range_portfolio]

    @jit
    def curve_cva(rhos):
        L = empty((len(rhos), Z_M.shape[0]))
        for r in range(0, len(rhos)):
            L[r, :] = calc_cva_merton(rhos[r])
        return L

    save_array(name_saved_file, curve_cva(rhos_merton))


    # TODO: finish that cva calculator for different values of rho (Loop).
    # TODO: Think about the cases where rho = 0 or rho = 1


def generate_cva_hull(Z_M, bs_hull, Q_default, name_saved_file, max_iter=100000, tol=1e-3):
    # need all times for the market factor, not only at exposure
    default_probabilities = Q_default(time_exposure)
    survival_probability = zeros(len(default_probabilities))
    survival_probability[0] = 1 - default_probabilities[0]
    for k in range(1, len(default_probabilities)):
        survival_probability[k] = survival_probability[k - 1] - default_probabilities[k]

    DiscountFactorXdefault = default_probabilities * discount_factor(time_exposure)

    # Newton Raphson fails to converge when b*Z_M are to big (b around 0.1 with Z_M aroud 500)
    # It should be because of one of a swap which has value around

    def calc_cva_hull(b):
        weightsHull = [
            Hull(b, Z=Z_M[k], timeline=timeline, survival_probability=survival_probability, times_default=time_default,
                 times_exposure=time_exposure, max_iter=max_iter, tol=tol) for k in range_portfolio]
        resultsDWR = array(
            [[sum(Z_M[k][index_exposure[t]] * weightsHull[k][t]) for t in range_exposure] for k in range_portfolio])
        return [sum(resultsDWR[k, :] * DiscountFactorXdefault) for k in range_portfolio]

    cva_hull = [calc_cva_hull(b) for b in bs_hull]
    save_array(name_saved_file, cva_hull)
