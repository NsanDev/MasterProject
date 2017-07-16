'''
Created on 2 avr. 2017

@author: Naitra
'''

# import matplotlib.pyplot as plt
from matplotlib.pyplot import get_current_fig_manager, plot, figure, show, savefig, xlabel, xlim, legend, grid
from numpy import array, linspace, vectorize
from Scripts.parameters import model, S0
import re
import pandas as pd
from datetime import datetime, date
if __name__ == '__main__':
    pass

###################
### Forward curve source https://www.barchart.com/futures/quotes/CL*0/all-futures?ref=excel#/viewName=main
###################
fig = figure(figsize=(8.0, 6.0))

forwardWTI = pd.read_csv("all-futures-contracts-intraday-07-14-2017.csv", usecols=["Contract", "Last"]).ix[1:]
forwardWTI['Contract'] = forwardWTI['Contract'].apply(lambda x: re.search('\((.*)\)', x).group(1).replace("'", "20"))
max_expiry = datetime(2020, 12, 1, 0, 0, 0)
forwardWTI['Expiry Date'] = forwardWTI['Contract'].apply(lambda x: datetime.strptime(x, '%b %Y'))
forwardWTI = forwardWTI[forwardWTI['Expiry Date'] <= '2021-01-01']
ax = forwardWTI.plot(x=['Expiry Date'], y=['Last'], grid=True, legend=False)  # , xticks=forwardWTI['Expiry Date'])
ax.set_ylabel("Price (USD)");
savefig('pictures/Current_Forward_curve.png', bbox_inches="tight")
###################
### Parameters
###################

# contract
T = 3
freq = 12
tenors = array([k / freq for k in range(0, int(T * freq) + 1)])

###################
### Simulation
###################

deltas = linspace(-0.2, 0.2, num=5, endpoint=True)

###################
### plots
###################

fig = figure(figsize=(8.0, 5.0))
for d_ini in deltas:
    fwd = vectorize(lambda T: model.forward(0, T, S0, d_ini))
    tenor_curve = fwd(tenors)
    plot(tenors, tenor_curve, label='$\delta=$' + str(d_ini))
xlabel('Delivery time (year)')
legend(ncol=1, loc='upper left')  #, bbox_to_anchor=(0, 0))
grid(linestyle='--', linewidth=1, axis='y')
xlim(xmin=0, xmax=max(tenors))
# show()
# figManager = get_current_fig_manager()
# figManager.window.showMaximized()
savefig('pictures/Forward_curve.png')
