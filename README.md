# DWR with weights approach 
This code has been written for my master's thesis. It contains an implementation of the Schwarz (1997) model and some class for introducing Directional Way Risk in the calculation of exposure's statistics.

## Getting started

### Installation
Just install python 3.6 with the standard packages from [Anaconda3](https://www.continuum.io/downloads). Here are some packages to add in your environment:
* numba
* statsmodel
* joblib

Running it on [PyCharm 2017.2.3 Community Edition](https://www.jetbrains.com/pycharm/download/) should make it work out of the box. I met some issues when I tried to run it in VSCode.

### A few words on the structure of the project

The *Scripts* folder contains the code that generates graphs and tables. 
* In particular, *parameters.py* contains all the constants for the models and the parameters of the simulations. 
* To generate the alpha surface in the folder *pictures*, just run the *main.py*. 
* To generate the results for the four selected instruments of the master's thesis, just run *statistics_exposure.py*. Note that each script take some time to run (around 10 minutes) because heavy calculations are made. 

I tried to introduce some speed up to reduce this timespan (multiprocessing on every cores of your computer + multithreading on each core (nb of threads = nb of cores).  Hence, I advice to close all opened programs before running it and stop any activity on your computer until the routine is done.

The other folders contain some tools for the scripts. Note that some of them are currently unused (such as *Calibration*).


