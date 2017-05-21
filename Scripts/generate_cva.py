from numpy import linspace

from Scripts.cva_calculator import generate_cva_indep, generate_cva_twofactors_hull
from Scripts.parameters import Q_default, save_array, load_array, load_all_array

S = load_array('spot_prices')
exposures = load_all_array('exposures*')
market_factor = load_all_array('Z*')

nb_rho_merton = 3
rhos_merton = [k / (nb_rho_merton + 1) for k in range(-nb_rho_merton, nb_rho_merton + 1)]
bS = linspace(start=-0.1, stop=0.1, num=3, endpoint=True)
bV = bS

############################
### Independent
############################
for key in exposures:
    generate_cva_indep(Exposures=exposures[key], Q_default=Q_default,
                       name_saved_file='cva_indep' + key.replace('exposures', '').replace('_', ''))
# generate_cva_indep(Exposures=exposures['exposures'], Q_default=Q_default, name_saved_file='cva_indep')

############################
### Merton
############################
# generate_cva_merton(exposures,rhos_merton, Q_default=Q_default, name_saved_file='cva_merton')

############################
### Hull
############################
generate_cva_twofactors_hull(Exposures=exposures['exposures'],
                             Z_M1=market_factor['Z_S'], bs1=bS,
                             Z_M2=market_factor['Z_V'], bs2=bV,
                             Q_default=Q_default, name_saved_file='cva_hull_two_fac', max_iter=100000, tol=1e-3)

# TODO: Add cva for greeks (Qdefault and S) but dont dave it! compute greek and save greek directly
# TODO: Add parameter to choose if array will be saved
# TODO: generate cva have to always return the array of cva!
save_array('rhos_merton', rhos_merton)
save_array('bS', bS)
save_array('bS', bV)

a = 1
