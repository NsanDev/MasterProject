'''
Created on 2 avr. 2017

@author: Naitra
'''

from scipy.stats import norm 
from math import log, sqrt, exp, pi

'''
Constant
'''
sqrtdpi = sqrt(2*pi)


def DistanceToDefault(S0,r,sigma,K,T,div=0):
    return ( log(S0/K) + (r-div-1/2*sigma*sigma)*T ) / (sigma*sqrt(T));

def DefaultProbability(S0,r,sigma,K,T,div=0):
    return norm.cdf(-DistanceToDefault(S0,r,sigma,K,T,div))

def Call(S0,r,sigma,K,T,div=0):
    if T == 0:
        return max(S0-K,0)
    else:
        d2 = DistanceToDefault(S0,r,sigma,K,T,div)
        d1 = d2 + sigma*sqrt(T)
        return S0*exp(-div*T)*norm.cdf(d1)-K*exp(-r*T)*norm.cdf(d2)

def Put(S0,r,sigma,K,T,div=0):
    return -S0*exp(-div*T) + K*exp(-r*T) + Call(S0,r,sigma,K,T,div)


def Call_Delta(S0,r,sigma,K,T,div=0):
    d1 = DistanceToDefault(S0,r,sigma,K,T,div) + sigma*sqrt(T)
    return exp(-div*T)*norm.cdf(d1)

def Call_Gamma(S0,r,sigma,K,T,div=0):
    d1 = DistanceToDefault(S0,r,sigma,K,T,div) + sigma*sqrt(T)
    return exp(-div*T)/(S0*sigma*sqrt(T)*sqrtdpi)*exp(-d1*d1/2)

def Call_Theta(S0,r,sigma,K,T,div=0):
    d2 = DistanceToDefault(S0,r,sigma,K,T,div)
    d1 = d2 + sigma*sqrt(T)
    disc_div = exp(-div*T)
    return -(S0*sigma*disc_div) / (2*sqrt(T)*sqrtdpi) * exp(-d1*d1/2) \
        -r*K*exp(-r*T)*norm.cdf(d2) + div*S0*disc_div*norm.cdf(d1)
        
def Call_Vega(S0,r,sigma,K,T,div=0):
    d2 = DistanceToDefault(S0,r,sigma,K,T,div)
    d1 = d2 + sigma*sqrt(T)
    return S0*exp(-div*T) * sqrt(T)/sqrtdpi * exp(-d1*d1/2)

def Call_Rho(S0,r,sigma,K,T,div=0):
    return K*T*exp(-r*T)*norm.cdf(DistanceToDefault(S0, r, sigma, K, T, div))

