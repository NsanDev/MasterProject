import statsmodels.api as sm
from matplotlib.pyplot import figure, xlabel, plot, ylabel, grid, xlim
from pandas import DataFrame

from Scripts.parameters import load_array

cva_indep = load_array('cva_indep')
cva_merton = load_array('cva_merton')
rhos_merton = load_array('rhos_merton')

alphas = []
pvalues = []
rsquared_adj = []
x = cva_indep

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
