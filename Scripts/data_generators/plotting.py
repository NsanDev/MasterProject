from matplotlib import cm
from matplotlib.pyplot import subplots, figure, legend
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numpy import meshgrid
from mpl_toolkits.mplot3d import Axes3D

from Scripts.parameters import save_figure, load_all_dataframe


def surface_hull(dataframe_hull, name_fig, angle=(25, 120)):
    # for surface plots
    allbs = sorted(set(dataframe_hull['b_S']))
    allbv = sorted(set(dataframe_hull['b_V']))
    alphas = dataframe_hull['alpha']
    X, Y = meshgrid(allbv, allbs)
    Z = alphas.values.reshape((len(allbs), len(allbv)))  # TODO check if the reshape/meshgrid is good
    fig = figure()

    ax = fig.gca(projection='3d')
    ax.view_init(*angle)
    surf = ax.plot_surface(X=X, Y=Y, Z=Z, cmap=cm.coolwarm, antialiased=True)
    # Customize the z axis.

    ax.set_ylabel(r'$b_S$', fontsize=14)
    ax.set_ylim(min(allbs), max(allbs))
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_rotate_label(False)
    ax.set_yticklabels([min(allbs), min(allbs) / 2, 0, max(allbs) / 2, max(allbs)], verticalalignment='baseline')

    ax.set_xlabel(r'$b_V$', fontsize=14)
    ax.set_xlim(min(allbv), max(allbv))
    ax.xaxis.set_major_locator(LinearLocator(5))
    ax.xaxis.set_rotate_label(False)
    ax.set_xticklabels([min(allbv), min(allbv) / 2, 0, max(allbv) / 2, max(allbv)], verticalalignment='baseline')

    ax.set_zlabel(r'$\alpha$', fontsize=14, )
    ax.zaxis.set_major_locator(LinearLocator(4))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.zaxis.set_rotate_label(False)

    fig.colorbar(surf, shrink=0.7, pad=0.1)  # aspect=5) # Add a color bar which maps values to colors.
    fig.tight_layout()
    save_figure('surface_' + name_fig, figure=fig)


def plot_hull(dataframe, name_fig):
    df = dataframe.loc[:, ['b_S', 'b_V', 'alpha', 'CI95_low', 'CI95_up']]

    fig, axes = subplots(nrows=3, ncols=2, sharex=True, sharey=False)

    allb = sorted(set(dataframe['b_S']))
    i = 0
    for k in [0, allb.index(0), len(allb) - 1]:
        ddf = df.loc[df['b_S'] == allb[k], ['b_V', 'alpha', 'CI95_low', 'CI95_up']]
        ax = ddf.plot(x='b_V', y=['alpha', 'CI95_low', 'CI95_up']
                      , color=['blue', 'red', 'red']
                      , style=['-', ':', ':']
                      , grid=True
                      , ax=axes[i][0]
                      , legend=False)
        ax.set(xlabel='$b_V$', ylabel=r'$b_S=$' + str(allb[k]))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        i = i + 1

    allb = sorted(set(dataframe['b_V']))
    i = 0
    for k in [0, allb.index(0), len(allb) - 1]:
        ddf = df.loc[df['b_V'] == allb[k], ['b_S', 'alpha', 'CI95_low', 'CI95_up']]
        ax = ddf.plot(x='b_S', y=['alpha', 'CI95_low', 'CI95_up']
                      , color=['blue', 'red', 'red']
                      , style=['-', ':', ':']
                      , grid=True
                      , ax=axes[i][1]
                      , legend=False)
        ax.set(xlabel='$b_S$', ylabel=r'$b_V=$' + str(allb[k]))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        i = i + 1

    legend([r'$\alpha$', r'$C.I._{95\%}$'], loc=(-0.35, -0.5), frameon=False)
    fig.tight_layout()
    save_figure('subplots_' + name_fig, figure=fig)
    # show()


if __name__ == '__main__':
    regression = load_all_dataframe('regression*')

    surface_hull(regression['regression_cva_hull'], 'cva_hull')
    surface_hull(regression['regression_delta_S'], 'regression_delta_S')
    surface_hull(regression['regression_delta_intensity'], 'regression_delta_intensity')

    plot_hull(regression['regression_cva_hull'], 'cva_hull')
    plot_hull(regression['regression_delta_S'], 'regression_delta_S')
    plot_hull(regression['regression_delta_intensity'], 'regression_delta_intensity')
