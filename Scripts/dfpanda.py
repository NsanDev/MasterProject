import os

from matplotlib.pyplot import subplots, figure, show, legend
from mpl_toolkits.mplot3d.axes3d import Axes3D
# from matplotlib.axes import Axes
from numpy import meshgrid
from pandas import read_pickle

Df = read_pickle('C:/Users/nthoun/Projects/MasterProject/Scripts/regression_result/regression_cva_hull.pkl')

df = Df.loc[:, ['b_S', 'b_V', 'alpha', 'CI95_low', 'CI95_up']]
allb = sorted(set(Df['b_S']))
# df.plot()
# fig = figure()
# ddf = df.loc[df['b_S'] == allb[0], ['b_V', 'alpha', 'CI95_low', 'CI95_up']]
# ddf.plot(x='b_V', y=['alpha'])
# show()
# from pandas import DataFrame
# from numpy import ones, array
# from matplotlib.pyplot import isinteractive
# isinteractive()
# data = ones((10,3))
# X = DataFrame(data=data, index=array(range(0,10)), columns=['c','b','a'])
# X.plot()

# for subplots only
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
legend([r'$\alpha$', r'$C.I._{95\%}$'], loc=(-0.1, 3.45))
show()
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter

# # for surface plots
# allbs = sorted(set(Df['b_S']))
# allbv = sorted(set(Df['b_V']))
# alphas = df['alpha']
# X, Y = meshgrid(allbs, allbv)
# Z = alphas.values.reshape((len(allbs), len(allbv)))
# fig = figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X=X, Y=Y, Z=Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# # Customize the z axis.
# ax.set_xlabel(r'$b_S$')
# ax.set_xlim(min(allbs), max(allbs))
# ax.xaxis.set_major_locator(LinearLocator(len(allbs)))
#
# ax.set_ylabel(r'$b_V$')
# ax.set_ylim(min(allbv), max(allbv))
# ax.yaxis.set_major_locator(LinearLocator(len(allbv)))
#
# ax.set_zlabel(r'$\alpha$')
# ax.zaxis.set_major_locator(LinearLocator(4))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
# fig.colorbar(surf, shrink=0.5, aspect=5)  # Add a color bar which maps values to colors.
