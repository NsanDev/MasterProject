from Scripts.parameters import load_all_array, Q_default, bS, bV
from time import clock
from Scripts.data_generators.cva_calculator import generate_cva_generalized_hull, generate_cvas
from Scripts.parameters import discount_factor, load_array
from joblib import Parallel, delayed
import multiprocessing

if __name__ == '__main__':
    exposures = load_all_array('exposures*')
    market_factor = load_all_array('Z*')

    start = clock()

    # def generate_cva_twofactors_hull(Exposures, Z_M1, Z_M2, bs1, bs2, Q_default, name_saved_file='', max_iter=100000,
    #                                  tol=1e-3):
    #
    #     cva = [Parallel(n_jobs=5, backend="threading")(delayed(
    #         generate_cva_generalized_hull)(Exposures, b1 * Z_M1 + b2 * Z_M2,
    #                                        Q_default, max_iter=max_iter, tol=tol) for b2 in bs2)
    #         for b1 in bs1]
    #
    #     # cva = [[generate_cva_generalized_hull(Exposures, b1 * Z_M1 + b2 * Z_M2,Q_default, max_iter=max_iter, tol=tol)
    #     #         for b2 in bs2] for b1 in bs1]
    #
    # num_cores = multiprocessing.cpu_count()
    # #Parallel(n_jobs=2,backend='threading')(delayed(job)() for i in range(0,3))
    # generate_cva_twofactors_hull(Exposures=exposures['exposures'],
    #                              Z_M1=market_factor['Z_S'], bs1=bS,
    #                              Z_M2=market_factor['Z_V'], bs2=bV,
    #                              Q_default=Q_default, name_saved_file='cva_hull', max_iter=100000, tol=1e-3)

    Parallel(n_jobs=4)(delayed(generate_cvas)(i) for i in range(1, 6))
    total = clock() - start

    a = 1
