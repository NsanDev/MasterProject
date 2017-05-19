from numpy import random, transpose, linspace, array, exp, save, load, maximum

from Scripts.portfolio import create_contracts
from StochasticProcess.Commodities.Schwartz97 import Schwartz97

##############################
### Parameters for Schwartz
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
### MC Simulation
##############################
Collateral_level = lambda MtM_value: 0  ## No collateral

##############################
### Parameters for hazard rate
##############################
start_default = 0.1
nb_point_intensity = 5
constant_intensity = 0.001  # have to be calibrated from cds

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

default_extension = '.npy'
default_folder = 'data/'
##############################################################
### Constant and functions created from the parameters above.
##############################################################

model = Schwartz97(r=r, sigma_s=sigma_s, kappa=kappa, alpha_tilde=alpha, sigma_e=sigma_e, rho=corr)
random.seed(seed=seed)


def load_model():
    return model


def portfolio():
    book, cashflows_times, contract_name = create_contracts(model)
    T_horizon_CVA = cashflows_times[-1]
    time_exposure = linspace(min(start_exposure, cashflows_times[0]),
                             T_horizon_CVA, nb_point_exposure, endpoint=True)  # times at which we want exposure
    time_exposure = sorted(set(time_exposure) | set(cashflows_times))
    timeline = linspace(start_path, T_horizon_CVA, nb_point_path, endpoint=True)
    timeline = array(sorted(set(timeline) | set(time_exposure)))
    return book, time_exposure, timeline, contract_name


def simulate_path(timeline):
    return transpose(model.PathQ(S_ini=S0, delta_ini=delta0, timeline=timeline, nb_path=nb_simulation), (2, 1, 0))


def Q_default(time_exposure):
    probabilities = exp(-constant_intensity * time_exposure)
    probabilities[1:] = probabilities[:-1] - probabilities[1:]
    probabilities[0] = 1 - probabilities[0]
    return probabilities


def cumulated_Q_survival(time_exposure):
    return exp(-constant_intensity * time_exposure)


def discount_factor(t):
    return exp(-r * t)


def exposure_function(MtM_value):
    return maximum(MtM_value - Collateral_level(MtM_value), 0)


def save_array(name, data_array, extension=default_extension, folder=default_folder):
    save(folder + name + extension, data_array)


def load_array(name, extension=default_extension, folder=default_folder):
    return load(folder + name + extension)
