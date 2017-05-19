import statsmodels.api as sm
from matplotlib.pyplot import figure, xlabel, plot, ylabel, grid, xlim, ticklabel_format
from pandas import DataFrame

from Scripts.parameters import load_array

cva_independant = load_array('cva_indep')
cva_merton = load_array('cva_merton')
cva_hull = load_array('cva_hull')
rhos_merton = load_array('rhos_merton')
bs_hull = load_array('bs_hull')


def alpha_analysis(y, x, parameters, name_parameters, latex_name_parameters, name_fig):
    alphas = []
    pvalues = []
    rsquared_adj = []
    for k in range(0, y.shape[0]):
        model = sm.OLS(endog=y, exog=x)  # no intercept by default
        fitted = model.fit()
        alphas.append(*fitted.params)
        pvalues.append(*fitted.pvalues)
        rsquared_adj.append(fitted.rsquared_adj)

    df = DataFrame({name_parameters: parameters,
                    'alpha': alphas,
                    'p-value': pvalues,
                    'R_{adj}': rsquared_adj})
    df = df[['rho', 'alpha', 'R_{adj}', 'p-value']]
    latex_table = df.to_latex(index=False)

    fig = figure()
    plot(parameters, alphas)
    xlabel(latex_name_parameters, fontsize=14)
    ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ylabel(r'$\alpha$', fontsize=14)
    grid(True, linestyle='--')
    xlim(xmin=min(parameters), xmax=max(parameters))
    fig.savefig('..\pictures\cva' + '\\' + name_fig + '.png')

    return df


regression_merton = alpha_analysis(cva_merton, cva_merton, rhos_merton, name_parameters='rho',
                                   latex_name_parameters=r'$\rho$', name_fig='alphas_merton2')
regression_hull = alpha_analysis(cva_hull, cva_independant, bs_hull, name_parameters='b',
                                 latex_name_parameters=r'$b$', name_fig='alphas_hull2')
