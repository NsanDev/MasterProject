from matplotlib import cm
from matplotlib.pyplot import figure, subplots, figure, show, legend
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy import meshgrid
from pandas import read_pickle

from Scripts.parameters import save_figure, load_all_dataframe


def surface_hull(dataframe_hull, name_fig):
    # for surface plots
    allbs = sorted(set(dataframe_hull['b_S']))
    allbv = sorted(set(dataframe_hull['b_V']))
    alphas = dataframe_hull['alpha']
    X, Y = meshgrid(allbv, allbs)
    Z = alphas.values.reshape((len(allbs), len(allbv)))  # TODO check if the reshape/meshgrid is good
    fig = figure()

    ax = fig.gca(projection='3d')
    ax.view_init(25, 120)
    surf = ax.plot_surface(X=X, Y=Y, Z=Z, cmap=cm.coolwarm, antialiased=True)
    # Customize the z axis.
    ax.set_ylabel(r'$b_S$', fontsize=14)
    # ax.set_ylim(min(allbs), max(allbs))
    ax.yaxis.set_major_locator(LinearLocator(len(allbs)))

    ax.set_xlabel(r'$b_V$', fontsize=14)
    # ax.set_xlim(min(allbv), max(allbv))
    ax.xaxis.set_major_locator(LinearLocator(len(allbv)))

    ax.set_zlabel(r'$\alpha$', fontsize=14)
    ax.zaxis.set_major_locator(LinearLocator(4))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.7, pad=0.1)  # aspect=5) # Add a color bar which maps values to colors.
    fig.tight_layout()
    save_figure('surface_' + name_fig, figure=fig)


def plot_hull(dataframe, name_fig):
    df = dataframe.loc[:, ['b_S', 'b_V', 'alpha', 'CI95_low', 'CI95_up']]
    allb = sorted(set(dataframe['b_S']))
    fig, axes = subplots(nrows=len(allb), ncols=1, sharex=True, sharey=False)
    for k in range(0, len(allb)):
        ddf = df.loc[df['b_S'] == allb[k], ['b_V', 'alpha', 'CI95_low', 'CI95_up']]
        ax = ddf.plot(x='b_V', y=['alpha', 'CI95_low', 'CI95_up']
                      , color=['blue', 'red', 'red']
                      , grid=True
                      , ax=axes[k]
                      , legend=False)
        ax.set(xlabel='$b_V$', ylabel=r'$b_S=$' + str(allb[k]))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    legend([r'$\alpha$', r'$C.I._{95\%}$'], loc=(0.01, 2.9))
    fig.tight_layout()
    save_figure('subplots_' + name_fig, figure=fig)
    # show()

regression = load_all_dataframe('regression*')

surface_hull(regression['regression_cva_hull'], 'cva_hull')
#surface_hull(regression['regression_delta_S'], 'regression_delta_S')
surface_hull(regression['regression_delta_intensity'], 'regression_delta_intensity')

plot_hull(regression['regression_cva_hull'], 'cva_hull')
# plot_hull(regression['regression_delta_S'], 'regression_delta_S')
plot_hull(regression['regression_delta_intensity'], 'regression_delta_intensity')
