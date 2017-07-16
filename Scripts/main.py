from time import clock
from Scripts.data_generators.simulations import launch_simulation
from joblib import Parallel, delayed
import multiprocessing
from Scripts.parameters import load_all_array, S0, shift_S, bS, bV

if __name__ == '__main__':
    start = clock()

    # Simulation
    launch_simulation()
    launch_simulation(S_ini=S0 + shift_S, shift_str='_shift_S_pos')
    launch_simulation(S_ini=S0 - shift_S, shift_str='_shift_S_neg')

    from Scripts.data_generators.cva_calculator import generate_cvas

    # Calcul cva
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(generate_cvas)(i) for i in range(1, 6))

    from Scripts.data_generators.alpha_analysis import alpha_analysis_hull

    # Regression
    cva = load_all_array('cva*')
    delta = load_all_array('delta*')
    regression_cva_hull = alpha_analysis_hull(cva['cva_hull'], cva['cva_indep'], bS=bS, bV=bV,
                                              name_dataframe='regression_cva_hull')
    regression_delta_S = alpha_analysis_hull(delta['delta_S_cva_hull'], delta['delta_S_cva_indep'], bS=bS, bV=bV,
                                             name_dataframe='regression_delta_S')
    regression_delta_intensity = alpha_analysis_hull(delta['delta_intensity_cva_hull'],
                                                     delta['delta_intensity_cva_indep'],
                                                     bS=bS, bV=bV, name_dataframe='regression_delta_intensity')

    from Scripts.data_generators.plotting import *

    # Plot directly saved
    regression = load_all_dataframe('regression*')
    surface_hull(regression['regression_cva_hull'], 'cva_hull', (15, -160))
    surface_hull(regression['regression_delta_S'], 'regression_delta_S', (15, -20))
    surface_hull(regression['regression_delta_intensity'], 'regression_delta_intensity', (15, -160))
    plot_hull(regression['regression_cva_hull'], 'cva_hull')
    plot_hull(regression['regression_delta_S'], 'regression_delta_S')
    plot_hull(regression['regression_delta_intensity'], 'regression_delta_intensity')

    total_time = clock() - start
    print(total_time)
