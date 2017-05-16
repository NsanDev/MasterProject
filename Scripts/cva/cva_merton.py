import statsmodels.api as sm
from matplotlib.pyplot import figure, xlabel, plot, ylabel, grid, xlim
from numpy import array
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

def calc_cva_merton(rho, Z_M=Z_M):
    weightsMerton = [[Merton(Z_M[k][t], rho, PD[t], tolerance=0.001) for t in range_exposure] for k in range_portfolio]
    resultsDWR = array([[sum(Z_M[k][t] * weightsMerton[k][t]) for t in range_exposure] for k in range_portfolio])
    return [sum(resultsDWR[k, :] * DiscountFactorXdefault) for k in range_portfolio]


cva_merton = [calc_cva_merton(rho, Z_M=Z_M) for rho in rhos_merton]
save_array('rhos_merton', rhos_merton)
save_array('cva_merton', cva_merton)

# TODO: finish that cva calculator for different values of rho (Loop).
    # TODO: Think about the cases where rho = 0 or rho = 1

alphas = []
pvalues = []
rsquared_adj = []
x = load_array('cva_indep')

for k in range(0, len(rhos_merton)):
    y = cva_merton[k]
    model = sm.OLS(endog=y, exog=x)  # no intercept by default
    fitted = model.fit()
    alphas.append(*fitted.params)
    pvalues.append(*fitted.pvalues)
    rsquared_adj.append(fitted.rsquared_adj)

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
