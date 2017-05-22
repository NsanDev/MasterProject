import os

from matplotlib.pyplot import subplots, figure
from numpy import meshgrid
from pandas import read_pickle

os.getcwd()

Df = read_pickle('C:/Users/nthoun/Projects/MasterProject/Scripts/regression_result/regression_cva_hull.pkl')

df = Df.loc[:, ['b_S', 'b_V', 'alpha', 'CI95_low', 'CI95_up']]
allb = sorted(set(Df['b_S']))

# for subplots only
fig, axes = subplots(nrows=3, ncols=1)
for k in range(0, len(allb)):
    ddf = df.loc[df['b_S'] == allb[k], ['b_V', 'alpha', 'CI95_low', 'CI95_up']]
    ax = ddf.plot(x='b_V', y=['alpha', 'CI95_low', 'CI95_up']
                  , title=r'$b_S=$' + str(allb[k])
                  , color=['blue', 'red', 'red']
                  , grid=True
                  , ax=axes[k])
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # ax.yaxis(r'$\alpha$', fontsize=14)
    ax.grid(True, linestyle='--')
    ax.set(xlabel=r'$b_V$', ylabel=r'$\alpha$')
    ax.legend([r'$\alpha$', r'$C.I._{95\%}$'])

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# for surface plots
allbs = sorted(set(Df['b_S']))
allbv = sorted(set(Df['b_V']))
alphas = df['alpha']
X, Y = meshgrid(allbs, allbv)
Z = alphas.values.reshape((len(allbs), len(allbv)))
fig = figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X=X, Y=Y, Z=Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_xlabel(r'$b_S$')
ax.set_xlim(min(allbs), max(allbs))
ax.xaxis.set_major_locator(LinearLocator(len(allbs)))

ax.set_ylabel(r'$b_V$')
ax.set_ylim(min(allbv), max(allbv))
ax.yaxis.set_major_locator(LinearLocator(len(allbv)))

ax.set_zlabel(r'$\alpha$')
ax.zaxis.set_major_locator(LinearLocator(4))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)  # Add a color bar which maps values to colors.
