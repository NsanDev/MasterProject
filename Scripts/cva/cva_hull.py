import statsmodels.api as sm
from matplotlib.pyplot import figure, xlabel, plot, ylabel, grid, xlim
from numpy import array, linspace
from pandas import DataFrame

from CreditModel.DirectionalWayRisk.Weights import Hull
from Scripts.parameters import Q_default, discount_factor, load_array, cumulated_Q_default

###################
### Load Data saved in folder data
###################
timeline = load_array('timeline')
time_exposure = load_array('time_exposure')
S = load_array('spot_prices')
convenience_yield = load_array('convenience_yields')
V_alltimes = load_array('contracts_alltimes')
Exposures_alltimes = load_array('exposures_alltimes')
index_exposure = [list(timeline).index(t) for t in time_exposure]
DiscountFactorXdefault = Q_default(time_exposure) * discount_factor(time_exposure)
range_exposure = range(0, len(time_exposure))
range_portfolio = range(0, len(V_alltimes))

time_default = array([timeline[-1]])
PD = cumulated_Q_default(time_default)

###################
### Choose Parameters on Hull
###################
Z_M = V_alltimes  # need all times, not only at exposure
bs_hull = linspace(start=-0.1, stop=0.1, num=30, endpoint=True)


###################
### Compute cva
###################

def calc_cva_hull(b, Z_M=Z_M):
    weightsHull = [Hull(b, Z=Z_M[k], timeline=timeline, probability_default=PD, times_default=time_default,
                        times_exposure=time_exposure) for k in range_portfolio]
    resultsDWR = array(
        [[sum(Z_M[k][index_exposure[t]] * weightsHull[k][t]) for t in range_exposure] for k in range_portfolio])
    return [sum(resultsDWR[k, :] * DiscountFactorXdefault) for k in range_portfolio]


cva_hull = [calc_cva_hull(b, Z_M=Z_M) for b in bs_hull]
# save_array('bs_hull', bs)
# save_array('cva_hull', cva_hull)
alphas = []
pvalues = []
rsquared_adj = []
x = load_array('cva_indep')

for k in range(0, len(bs_hull)):
    y = cva_hull[k]
    model = sm.OLS(endog=y, exog=x)  # no intercept by default
    fitted = model.fit()
    alphas.append(*fitted.params)
    pvalues.append(*fitted.pvalues)
    rsquared_adj.append(fitted.rsquared_adj)

df = DataFrame({'b': bs_hull,
                'alpha': alphas,
                'p-value': pvalues,
                'R_{adj}': rsquared_adj})
df = df[['b', 'alpha', 'R_{adj}', 'p-value']]
latex_table = df.to_latex(index=False)

fig = figure()
plot(bs_hull, alphas)
xlabel(r'$b$', fontsize=14)
ylabel(r'$\alpha$', fontsize=14)
grid(True, linestyle='--')
xlim(xmin=min(bs_hull), xmax=max(bs_hull))
fig.savefig(r'..\pictures\cva\alphas_hull.png')

# TODO: finish that cva calculator for different values of rho (Loop).
# TODO: Think about the cases where rho = 0 or rho = 1
