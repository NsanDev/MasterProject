from numpy import linspace, array_equal, allclose

from Scripts.cva_calculator import generate_cva_indep, generate_cva_hull
from Scripts.parameters import Q_default, save_array, load_array

timeline = load_array('timeline')
time_exposure = load_array('time_exposure')
S = load_array('spot_prices')
V_alltimes = load_array('contracts_alltimes')
exposures_alltimes = load_array('exposures_alltimes')
exposures = load_array('exposures')

rhos_merton = [k / 10 for k in range(-9, 9 + 1)]
bs_hull = linspace(start=-0.001, stop=0.001, num=11, endpoint=True)

cva_independant = generate_cva_indep(Q_default=Q_default, name_saved_file='cva_indep2')

# cva_merton = generate_cva_merton(exposures,rhos_merton, Q_default=Q_default, name_saved_file='cva_merton2')

cva_hull = generate_cva_hull(V_alltimes, bs_hull, Q_default, name_saved_file='cva_hull2',
                             max_iter=100000, tol=1e-3)

save_array('rhos_merton', rhos_merton)
save_array('bs_hull', bs_hull)

cva_indep = load_array('cva_indep')
cva_indep2 = load_array('cva_indep2')

cva_hull = load_array('cva_hull')
cva_hull2 = load_array('cva_hull2')

test1 = array_equal(cva_indep, cva_indep2)
test2 = allclose(cva_hull, cva_hull2)

a = 1
