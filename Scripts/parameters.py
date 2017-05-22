import glob

from numpy import random, transpose, linspace, array, exp, save, load, maximum
from pandas import read_pickle

from Common.Constant import eps
from Scripts.portfolio import create_contracts
from StochasticProcess.Commodities.Schwartz97 import Schwartz97

##############################
### Parameters of Schwartz97
##############################
S0 = 45
delta0 = 0.15
r = 0.1
sigma_s = 0.393
kappa = 1.876
sigma_e = 0.527
corr = 0.766
lamb = 0.198
alpha = 0.106 - lamb / kappa

##############################
### Collateral model
##############################
Collateral_level = lambda MtM_value: 0  ## No collateral

##############################
### Parameters for hazard rate
##############################
constant_intensity = 0.001  # have to be calibrated from cds

##############################
### Parameters of weights simulations
##############################
# Merton
nb_rho_merton = 3
rhos_merton = [k / (nb_rho_merton + 1) for k in range(-nb_rho_merton, nb_rho_merton + 1)]

# Hull
bS = linspace(start=-0.1, stop=0.1, num=3, endpoint=True)
bV = linspace(start=-0.1, stop=0.1, num=3, endpoint=True)

##############################
### Finite diff. for greeks/sensitivity
##############################
shift_S = 1E6 * eps  # sensi. of cva will be calculated with f(S_ini+shift_S)-f(S_ini-shift_S))/(2*shift_S)
shift_intensity = 1E6 * eps  # sensi. will be calculated with f(intensity+shift)-f(intensity-shift))/(2*shift_intensity)

##############################
### MC Simulation
##############################
nb_simulation = 1000
seed = 124

# time discretization of path
nb_point_path = 20
start_path = 0.01

# Times where exposure will be calculated
nb_point_exposure = 24
start_exposure = 0.05

default_extension_array = '.npy'
default_folder_array = 'data/'
default_extension_dataframe = '.pkl'
default_folder_dataframe = 'regression_result/'

##############################################################
### Constant and functions created from the parameters above.
##############################################################

model = Schwartz97(r=r, sigma_s=sigma_s, kappa=kappa, alpha_tilde=alpha, sigma_e=sigma_e, rho=corr)
random.seed(seed=seed)


def reset():
    random.seed(seed)


def load_model():
    return model


def portfolio():
    book, cashflows_times, contract_name = create_contracts(model, S0=S0)
    T_horizon_CVA = cashflows_times[-1]
    time_exposure = linspace(min(start_exposure, cashflows_times[0]),
                             T_horizon_CVA, nb_point_exposure, endpoint=True)  # times at which we want exposure
    time_exposure = sorted(set(time_exposure) | set(cashflows_times))
    timeline = linspace(start_path, T_horizon_CVA, nb_point_path, endpoint=True)
    timeline = array(sorted(set(timeline) | set(time_exposure)))
    return book, time_exposure, timeline, contract_name


def simulate_path(timeline, S_ini=S0, delta_ini=delta0, reset_seed=False):
    if reset_seed:
        reset()
    return transpose(model.PathQ(S_ini=S_ini, delta_ini=delta_ini, timeline=timeline, nb_path=nb_simulation), (2, 1, 0))


def Q_default(times, intensity=constant_intensity):
    probabilities = exp(-intensity * times)
    probabilities[1:] = probabilities[:-1] - probabilities[1:]
    probabilities[0] = 1 - probabilities[0]
    return probabilities


def Q_default_shift_pos(times):
    return Q_default(times=times, intensity=constant_intensity + shift_intensity)


def Q_default_shift_neg(times):
    return Q_default(times=times, intensity=constant_intensity - shift_intensity)


def cumulated_Q_survival(time_exposure):
    return exp(-constant_intensity * time_exposure)


def discount_factor(t):
    return exp(-r * t)


def exposure_function(MtM_value):
    return maximum(MtM_value - Collateral_level(MtM_value), 0)


def save_array(name, data_array, extension=default_extension_array, folder=default_folder_array):
    save(folder + name + extension, data_array)


def load_array(name, extension=default_extension_array, folder=default_folder_array):
    return load(folder + name + extension)


def load_all_array(name, folder=default_folder_array):
    result = {}
    mod_folder = folder.replace('/', '')
    for np_name in glob.glob(folder + name):
        result[np_name.replace('.npy', '').replace(mod_folder, '').replace('\\', '')] = load(np_name)
    return result


def save_dataframe(name, dataframe, folder=default_folder_dataframe):
    dataframe.to_pickle(folder + name + default_extension_dataframe)


def load_dataframe(name, folder=default_folder_dataframe):
    return read_pickle(folder + name + default_extension_dataframe)
