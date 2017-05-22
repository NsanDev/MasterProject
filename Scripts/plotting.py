from matplotlib import cm
from matplotlib.pyplot import figure
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numpy import meshgrid

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
    ax.set_xlabel(r'$b_S$')
    # ax.set_xlim(min(allbs), max(allbs))
    ax.xaxis.set_major_locator(LinearLocator(len(allbs)))

    ax.set_ylabel(r'$b_V$')
    # ax.set_ylim(min(allbv), max(allbv))
    ax.yaxis.set_major_locator(LinearLocator(len(allbv)))

    ax.set_zlabel(r'$\alpha$')
    ax.zaxis.set_major_locator(LinearLocator(4))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.7, pad=0.1)  # aspect=5) # Add a color bar which maps values to colors.
    save_figure(name_fig, figure=fig)


regression = load_all_dataframe('regression*')

surface_hull(regression['regression_cva_hull'], 'cva_hull')
surface_hull(regression['regression_delta_S'], 'regression_delta_S')
surface_hull(regression['regression_delta_intensity'], 'regression_delta_intensity')
