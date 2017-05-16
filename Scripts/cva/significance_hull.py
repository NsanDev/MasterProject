import statsmodels.api as sm
from matplotlib.pyplot import figure, xlabel, plot, ylabel, grid, xlim
from pandas import DataFrame

from Scripts.data_generators.parameters import load_array

cva_indep = load_array('cva_indep')
cva_hull = load_array('cva_hull')
bs_hull = load_array('bs_hull')

alphas = []
pvalues = []
rsquared_adj = []
x = cva_indep

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
