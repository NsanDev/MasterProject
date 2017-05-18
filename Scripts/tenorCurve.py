'''
Created on 2 avr. 2017

@author: Naitra
'''

# import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from numpy import array, linspace, vectorize

from StochasticProcess.Commodities.Schwartz97 import Schwartz97

if __name__ == '__main__':
    pass

###################
### Parameters
###################

# parameters schwartz
S0 = 45
delta0 = 0.15
r = 0.02

sigma_s = 0.393
kappa = 1.876
sigma_e = 0.527
corr = 0.766
lamb = 0.198
alpha = 0.106 - lamb / kappa
# contract
T = 3
freq = 12
tenors = array([k / freq for k in range(0, int(T * freq) + 1)])

###################
### Simulation
###################

model = Schwartz97(r=r, sigma_s=sigma_s, kappa=kappa, alpha_tilde=alpha, sigma_e=sigma_e, rho=corr)
deltas = linspace(-0.2, 0.2, num=5, endpoint=True)

###################
### plots
###################

fig = figure()
for d_ini in deltas:  # 3 is the number of parameters
    fwd = vectorize(lambda T: model.forward(0, T, S0, d_ini))
    tenor_curve = fwd(tenors)
    plot(tenors, tenor_curve, label='$\delta=$' + str(d_ini))
xlabel('Delivery time (year)')
legend(ncol=1, loc='lower left', bbox_to_anchor=(0, 0))
grid(linestyle='--', linewidth=1, axis='y')
xlim(xmin=0, xmax=max(tenors))
# show()
savefig('pictures/Forward_curve.png')
