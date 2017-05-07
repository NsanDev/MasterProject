from scipy.stats import norm
from numpy import log, sqrt, exp, maximum



'''
output: E(exp(X)-K) 
X = N(mu,sigmasq)
K is constant
'''
def Call(mu, sigmasq, K):
    if sigmasq == 0:
        return maximum(exp(mu)-K, 0)
    else:
        return exp(mu+0.5*sigmasq)*norm.cdf((mu+sigmasq-log(K))/sqrt(sigmasq)) - \
               K*norm.cdf((mu-log(K))/sqrt(sigmasq))

def Put(mu, sigma, K):
    return exp(mu+0.5*sigma**2)- K - Call(mu,sigma,K)

def Magrabe(mu1, sigmasq1, mu2, sigmasq2, cov):
    mu = mu1-mu2-sigmasq2+cov
    sigmasq = sigmasq1 + sigmasq2 - 2*cov
    return exp(mu1+sigmasq1)*norm.cdf((mu+sigmasq)/sqrt(sigmasq)) - \
           exp(mu2+sigmasq2)*norm.cdf(mu/sqrt(sigmasq))