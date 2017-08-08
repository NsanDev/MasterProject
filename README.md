# Getting started

A python 3.6 installation with the standard packages from Anaconda3 to run this code. 

The *Scripts* folder contains the code that generates graphs and tables. In particular, *parameters.py* contains all the constants for the models and the parameters of the simulations. To generate the alpha surface in the folder *pictures*, just run the *main.py*. To generate the results for the four selected instruments of the master's thesis, just run *statistics_exposure.py*. Note that each script take some time to run (around 10 minutes) because heavy calculations are made. I tried to introduce some speed up to reduce this time (multiprocessing on every cores of your computer + multithreading on each core (nb of threads = nb of cores).  Hence, I advice to close all opened programs before running it and stop any activity on your computer until the routine is done.

The other folders contain some tools for the scripts. Note that some of them are currently unused (such as *Calibration*).


