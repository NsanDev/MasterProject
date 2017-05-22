from Scripts.cva_calculator import generate_cva_indep, generate_cva_twofactors_hull
from Scripts.parameters import Q_default, save_array, load_all_array, shift_S, \
    Q_default_shift_neg, Q_default_shift_pos, \
    shift_intensity, bS, bV

# This class do not need to know that cst intensity model is used

exposures = load_all_array('exposures*')
market_factor = load_all_array('Z*')

############################
### Independent
############################

generate_cva_indep(Exposures=exposures['exposures'], Q_default=Q_default, name_saved_file='cva_indep')

cva_indep_shift_S_pos = generate_cva_indep(Exposures=exposures['exposures_shift_S_pos'], Q_default=Q_default)
cva_indep_shift_S_neg = generate_cva_indep(Exposures=exposures['exposures_shift_S_neg'], Q_default=Q_default)
delta_S_cva_indep = (cva_indep_shift_S_pos - cva_indep_shift_S_neg) / (2 * shift_S)
save_array('delta_S_cva_indep', delta_S_cva_indep)

cva_indep_shift_intensity_pos = generate_cva_indep(Exposures=exposures['exposures'], Q_default=Q_default_shift_pos)
cva_indep_shift_intensity_neg = generate_cva_indep(Exposures=exposures['exposures'], Q_default=Q_default_shift_neg)
delta_intensity_cva_indep = (cva_indep_shift_intensity_pos - cva_indep_shift_intensity_neg) \
                            / (2 * shift_intensity)
save_array('delta_intensity_cva_indep', delta_intensity_cva_indep)

# generate_cva_indep(Exposures=exposures['exposures'], Q_default=Q_default, name_saved_file='cva_indep')

############################
### Merton
############################
# generate_cva_merton(exposures,rhos_merton, Q_default=Q_default, name_saved_file='cva_merton')

############################
### Hull
############################

# CVA
generate_cva_twofactors_hull(Exposures=exposures['exposures'],
                             Z_M1=market_factor['Z_S'], bs1=bS,
                             Z_M2=market_factor['Z_V'], bs2=bV,
                             Q_default=Q_default, name_saved_file='cva_hull', max_iter=100000, tol=1e-3)

# delta_S
cva_hull_shift_S_pos = generate_cva_twofactors_hull(Exposures=exposures['exposures_shift_S_pos'],
                                                    Z_M1=market_factor['Z_S_shift_S_pos'], bs1=bS,
                                                    Z_M2=market_factor['Z_V_shift_S_pos'], bs2=bV,
                                                    Q_default=Q_default, max_iter=100000, tol=1e-3)
cva_hull_shift_S_neg = generate_cva_twofactors_hull(Exposures=exposures['exposures_shift_S_neg'],
                                                    Z_M1=market_factor['Z_S_shift_S_neg'], bs1=bS,
                                                    Z_M2=market_factor['Z_V_shift_S_neg'], bs2=bV,
                                                    Q_default=Q_default, max_iter=100000, tol=1e-3)

delta_S_cva_hull = (cva_hull_shift_S_pos - cva_hull_shift_S_neg) / (2 * shift_S)
save_array('delta_S_cva_hull', delta_S_cva_hull)

# delta_intensity
cva_hull_shift_intensity_pos = generate_cva_twofactors_hull(Exposures=exposures['exposures'],
                                                            Z_M1=market_factor['Z_S'], bs1=bS,
                                                            Z_M2=market_factor['Z_V'], bs2=bV,
                                                            Q_default=Q_default_shift_pos, max_iter=100000, tol=1e-3)

cva_hull_shift_intensity_neg = generate_cva_twofactors_hull(Exposures=exposures['exposures'],
                                                            Z_M1=market_factor['Z_S'], bs1=bS,
                                                            Z_M2=market_factor['Z_V'], bs2=bV,
                                                            Q_default=Q_default_shift_neg, max_iter=100000, tol=1e-3)
delta_intensity_cva_hull = (cva_hull_shift_intensity_pos - cva_hull_shift_intensity_neg) / (2 * shift_intensity)
save_array('delta_intensity_cva_hull', delta_intensity_cva_hull)

a = 1
