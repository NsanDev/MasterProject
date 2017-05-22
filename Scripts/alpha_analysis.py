from matplotlib.pyplot import figure, xlabel, plot, ylabel, grid, xlim, ticklabel_format, legend
from numpy import array
from pandas import DataFrame
from scipy.stats import t
from statsmodels.api import OLS  # for regression

from Scripts.parameters import load_array, load_all_array, \
    bS, bV

cva = load_all_array('cva*')
delta = load_all_array('delta*')
# cva_merton = load_array('cva_merton')
cva_hull = load_array('cva_hull')


# plot_surface(X, Y, Z, *args, **kwargs)


def alpha_analysis(y, x, parameters, name_parameters, latex_name_parameters, name_fig, CI=True):
    alphas = []
    pvalues = []
    rsquared_adj = []
    s = []
    for k in range(0, y.shape[0]):
        model = OLS(endog=y[k], exog=x)  # no intercept by default
        fitted = model.fit()
        alphas.append(*fitted.params)
        pvalues.append(*fitted.pvalues)
        rsquared_adj.append(fitted.rsquared_adj)
        s.append(fitted.cov_HC0[0, 0])

    df = DataFrame({name_parameters: parameters,
                    'alpha': alphas,
                    'p-value': pvalues,
                    'R_{adj}': rsquared_adj})
    df = df[[name_parameters, 'alpha', 'R_{adj}', 'p-value']]
    # latex_table = df.to_latex(index=False)

    alphas = array(alphas)
    s = array(s)
    fig = figure()

    plot(parameters, alphas, 'blue', label=r'$\alpha$')
    if CI:
        CI_up = alphas + t.ppf(0.975, len(x) - 1) * s
        CI_low = alphas - t.ppf(0.975, len(x) - 1) * s
        plot(parameters, CI_low, color='red')
        plot(parameters, CI_up, color='red', label='95% CI')
    legend()
    xlabel(latex_name_parameters, fontsize=14)
    ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ylabel(r'$\alpha$', fontsize=14)
    grid(True, linestyle='--')
    xlim(xmin=min(parameters), xmax=max(parameters))
    fig.savefig('pictures\cva' + '\\' + name_fig + '.png')

    return df


def alpha_analysis_hull(y, x, bS, bV):
    alphas = []
    pvalues = []
    rsquared_adj = []
    CI_up = []
    CI_low = []
    duplicate_b_S = []
    duplicate_b_V = []
    s = []
    for k in range(0, y.shape[0]):
        for l in range(0, y.shape[1]):
            model = OLS(endog=y[k, l,], exog=x)  # no intercept by default
            fitted = model.fit()
            alphas.append(*fitted.params)
            pvalues.append(*fitted.pvalues)
            rsquared_adj.append(fitted.rsquared_adj)
            s.append(fitted.cov_HC0[0, 0])
            duplicate_b_S.append(bS[k])
            duplicate_b_V.append(bV[l])
    s = array(s)
    CI_up = (alphas + t.ppf(0.975, len(x) - 1) * s)
    CI_low = (alphas - t.ppf(0.975, len(x) - 1) * s)
    df = DataFrame({'b_S': duplicate_b_S,
                    'b_V': duplicate_b_V,
                    'alpha': alphas,
                    'Standard Error': s,
                    'p-value': pvalues,
                    'R_{adj}': rsquared_adj,
                    'CI95%_up': CI_up,
                    'CI95%_low': CI_low})
    df = df[['b_S', 'b_V', 'alpha', 'Standard Error', 'R_{adj}', 'p-value', 'CI95%_low', 'CI95%_up']]
    # latex_table = df.to_latex(index=False)
    return df

# regression_merton = alpha_analysis(cva_merton, cva_independant, rhos_merton, name_parameters='rho',
#                                   latex_name_parameters=r'$\rho$', name_fig='alphas_merton')

# regression_hull = alpha_analysis(cva_hull, cva_independant, bs_hull, name_parameters='b',
#                                 latex_name_parameters=r'$b$', name_fig='alphas_hull', CI=True)

regression_cva_hull = alpha_analysis_hull(cva['cva_hull'], cva['cva_indep'], bS=bS, bV=bV)
regression_cva_hull.to_pickle('regression_result/regression_cva_hull.pkl')
regression_delta_S = alpha_analysis_hull(delta['delta_S_cva_hull'], delta['delta_S_cva_indep'], bS=bS, bV=bV)
regression_delta_S.to_pickle('regression_result/regression_delta_S.pkl')
regression_delta_intensity = alpha_analysis_hull(delta['delta_intensity_cva_hull'], delta['delta_intensity_cva_indep'],
                                                 bS=bS, bV=bV)
regression_delta_intensity.to_pickle('regression_result/regression_delta_intensity.pkl')
a = 1
