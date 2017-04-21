from numpy import array, zeros, exp, log, sqrt
from numpy.random import multivariate_normal
'''
Documentation in:
https://cran.r-project.org/web/packages/schwartz97/vignettes/TechnicalDocument.pdf
'''

class Schwartz97:


    def __init__(self, S0, delta0, mu, sigma_s, kappa, alpha, sigma_e, rho):
        self.S0 = S0
        self.delta0 = delta0
        self.mu = mu
        self.sigma_s = sigma_s
        self.kappa = kappa
        self.alpha = alpha
        self.sigma_e = sigma_e
        self.rho = rho

    ##############################################
    ### Path Generator
    ##############################################

    def Path(self, timeline, nb_path, spot_only=False):

        dsigmas = 0.5*self.sigma_s**2
        adjused_drift = (self.mu-dsigmas)
        T = len(timeline)
        L = zeros([nb_path, T,2])
        W = multivariate_normal([0,0], [[1,self.rho],[self.rho,1]], (nb_path,T))

        L[:, 0, 0] = log(self.S0) + (adjused_drift-self.delta0)*timeline[0] + self.sigma_s*sqrt(timeline[0])*W[:, 0, 0]
        L[:, 0, 1] = self.delta0 + self.kappa*(self.alpha-self.delta0)*timeline[0] + self.sigma_e*sqrt(timeline[0])*W[:, 0, 1]

        for t in range(1, T):
            delta = (timeline[t] - timeline[t - 1])
            sqrtdelta = sqrt(delta)
            L[:, t, 0] = L[:, t-1, 0] + self.kappa*(adjused_drift-L[:, t-1, 1])*delta + self.sigma_s*sqrtdelta*W[:, t, 0]
            L[:, t, 1] = L[:, t-1, 1] + (self.alpha-L[:, t-1, 1])*delta + self.sigma_e*sqrtdelta*W[:, t, 1]

        L[:, :, 0] = exp(L[:, :, 0])
        if spot_only:
            return L[:, :, 0]
        else:
            return L


    ##############################################
    ### Joint distribution of State Variable
    ##############################################

    def A(self, T):
        cov = self.sigma_s * self.sigma_e * self.rho
        sigma_delta_sq = self.sigma_e ** 2
        kappa_delta_sq = self.kappa ** 2
        result = (self.mu - self.alpha + 0.5 * sigma_delta_sq / kappa_delta_sq - (cov / self.kappa)) * T \
            + 0.25 * sigma_delta_sq * (1 - exp(-2 * self.kappa * T)) / (self.kappa ** 3) \
            + (self.kappa * self.alpha + cov - sigma_delta_sq / self.kappa) \
            * (1 - exp(-self.kappa * T)) / self.kappa / self.kappa
        return result

    def B(self,T):
        return - (1 - exp(-self.kappa * T)) / self.kappa

    def mu_X (self,T):
        return log(self.S0) + (self.mu - 0.5*self.sigma_s**2-self.alpha)*T+(self.alpha-self.delta0)*(1 - exp(-self.kappa))/self.kappa

    def mu_delta(self,T):
        return exp(-self.kappa*T)*self.delta0 + self.alpha*(1 - exp(-self.kappa))

    def sigma2_X(self,T):
        return (self.sigma_e/self.kappa)**2*((1 - exp(-2*self.kappa))/(2*self.kappa)-2*(1 - exp(-self.kappa))/self.kappa + T)+2*self.sigma_s*self.sigma_e*self.rho/self.kappa*((1 - exp(-self.kappa))/self.kappa-T) + self.sigma_s**2*T

    def sigma2_delta(self,T):
        return self.sigma_e**2/self.kappa*(1-exp(-2*self.kappa*T))

    def sigma_Xdelta(self,T):
        return 1/self.kappa*((self.sigma_s*self.sigma_e*self.rho-self.sigma_e**2/self.kappa)*(1-exp(-self.kappa*T))+self.sigma_e**2/(2*self.kappa)*(1 - exp(-2*self.kappa*T)))

    def G(self,T):
        return exp(self.mu_X(T)+0.5*self.sigma2_X(T))


    ####################
    ### getter/setter
    ####################

    def getter(self):
        return (self.mu, self.sigma_s, self.kappa, self.alpha, self.sigma_e, self.rho)

    def setter(self, mu_s, sigma_s, kappa_delta, mu_delta, sigma_delta, corr):
        self.mu = mu_s
        self.sigma_s = sigma_s
        self.kappa = kappa_delta
        self.alpha = mu_delta
        self.sigma_e = sigma_delta
        self.rho = corr

    ####################
    ### properties
    ####################

    parameters = property(getter, setter)

