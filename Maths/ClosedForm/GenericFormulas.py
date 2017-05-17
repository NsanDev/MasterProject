from numpy import log, sqrt, exp, maximum
from scipy.stats import norm

'''
output: E[max(exp(X)-K,0)] 
X = N(mu,sigmasq)
K is constant
'''
def Call(mu, sigmasq, K):
    if sigmasq == 0:
        return maximum(exp(mu)-K, 0)
    else:
        return exp(mu + 0.5 * sigmasq) * norm.cdf((mu + sigmasq - log(K)) / sqrt(sigmasq)) - K * norm.cdf(
            (mu - log(K)) / sqrt(sigmasq))


'''
output: E[max(K-exp(X),0)] 
X = N(mu,sigmasq)
K is constant
'''
def Put(mu, sigmasq, K):
    return Call(mu, sigmasq, K) - (exp(mu + 0.5 * sigmasq) - K)  # Call-Put parity


'''
output: E[max(exp(X1)-exp(X2),0)] 
X1 = N(mu1,sigmasq1)
X2 = N(mu2,sigmasq2)
'''
def Magrabe(mu1, sigmasq1, mu2, sigmasq2, cov):
    mu = mu1-mu2-sigmasq2+cov
    sigmasq = sigmasq1 + sigmasq2 - 2*cov
    return exp(mu1 + sigmasq1) * norm.cdf((mu + sigmasq) / sqrt(sigmasq)) - exp(mu2 + sigmasq2) * norm.cdf(
        mu / sqrt(sigmasq))
