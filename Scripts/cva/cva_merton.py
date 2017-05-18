from time import clock

import statsmodels.api as sm
from matplotlib.pyplot import figure, xlabel, plot, ylabel, grid, xlim
from numba import jit
from numpy import array, zeros, empty
from pandas import DataFrame

from CreditModel.DirectionalWayRisk.Weights import Merton
from Scripts.parameters import Q_default, discount_factor, save_array, load_array

###################
### Load Data saved in folder data
###################
timeline = load_array('timeline')
time_exposure = load_array('time_exposure')
S = load_array('spot_prices')
convenience_yield = load_array('convenience_yields')
V = load_array('contracts')
Exposures = load_array('exposures')
DiscountFactorXdefault = Q_default(time_exposure) * discount_factor(time_exposure)
range_exposure = range(0, len(time_exposure))
range_portfolio = range(0, len(V))
PD = Q_default(time_exposure)

###################
### Choose Parameters on merton
###################
Z_M = Exposures
rhos_merton = [k / 10 for k in range(-9, 9 + 1)]


###################
### Compute cva
###################

@jit
def calc_cva_merton(rho, Z_M=Z_M):
    result = zeros(Z_M.shape[0])
    for k in range_portfolio:
        for t in range_exposure:
            result[k] = result[k] + DiscountFactorXdefault[t] * \
                                    sum(Z_M[k][t] * Merton(Z_M[k][t], rho, PD[t], tolerance=0.001))
    return result


def calc_cva_merton0(rho, Z_M=Z_M):
    weightsMerton = [[Merton(Z_M[k][t], rho, PD[t], tolerance=0.001) for t in range_exposure] for k in range_portfolio]
    resultsDWR = array([[sum(Z_M[k][t] * weightsMerton[k][t]) for t in range_exposure] for k in range_portfolio])
    return [sum(resultsDWR[k, :] * DiscountFactorXdefault) for k in range_portfolio]


@jit
def curve_cva(rhos, Z_M=Z_M):
    L = empty((len(rhos), Z_M.shape[0]))
    for r in range(0, len(rhos)):
        L[r, :] = calc_cva_merton(rhos[r])
    return L


t = clock()
cva_merton = curve_cva(rhos_merton)
t = clock() - t
# cva_merton = [calc_cva_merton0(rho, Z_M=Z_M) for rho in rhos_merton]
save_array('rhos_merton', rhos_merton)
save_array('cva_merton', cva_merton)

# TODO: finish that cva calculator for different values of rho (Loop).
# TODO: Think about the cases where rho = 0 or rho = 1

alphas = []
pvalues = []
rsquared_adj = []
x = load_array('cva_indep')


@jit
def regression():
    for k in range(0, len(rhos_merton)):
        y = cva_merton[k]
        model = sm.OLS(endog=y, exog=x)  # no intercept by default
        fitted = model.fit()
        alphas.append(*fitted.params)
        pvalues.append(*fitted.pvalues)
        rsquared_adj.append(fitted.rsquared_adj)


regression()

df = DataFrame({'rho': rhos_merton,
                'alpha': alphas,
                'p-value': pvalues,
                'R_{adj}': rsquared_adj})
df = df[['rho', 'alpha', 'R_{adj}', 'p-value']]
latex_table = df.to_latex(index=False)

fig = figure()
plot(rhos_merton, alphas)
xlabel(r'$\rho$', fontsize=14)
ylabel(r'$\alpha$', fontsize=14)
grid(True, linestyle='--')
xlim(xmin=min(rhos_merton), xmax=max(rhos_merton))
fig.savefig(r'..\pictures\cva\alphas_merton.png')
