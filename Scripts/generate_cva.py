from time import clock

from numpy import linspace

from Scripts.cva_calculator import generate_cva_hull
from Scripts.parameters import Q_default, save_array, load_array

S = load_array('spot_prices')
exposures = load_array('exposures')
V_alltimes = load_array('contracts_alltimes')
exposures_alltimes = load_array('exposures_alltimes')

nb_rho_merton = 5
rhos_merton = [k / (nb_rho_merton + 1) for k in range(-nb_rho_merton, nb_rho_merton + 1)]
bs_hull = linspace(start=-0.03, stop=0.03, num=5, endpoint=True)

t = clock()
# generate_cva_indep(Q_default=Q_default, name_saved_file='cva_indep')
# generate_cva_merton(exposures,rhos_merton, Q_default=Q_default, name_saved_file='cva_merton')
generate_cva_hull(V_alltimes, bs_hull, Q_default, name_saved_file='cva_hull',
                  max_iter=10000000, tol=1e-3)
t = clock() - t
# generate_cva_hull([V_alltimes, bs_hull, Q_default, name_saved_file='cva_hull',
#                             max_iter=10000000, tol=1e-3)
save_array('rhos_merton', rhos_merton)
save_array('bs_hull', bs_hull)

a = 1
