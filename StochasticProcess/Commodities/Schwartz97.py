from numpy import array, zeros, exp, log, sqrt, maximum, less_equal
from numpy.random import multivariate_normal
from bisect import bisect_left
from Maths.ClosedForm.GenericFormulas import Call as generic_call

'''
Documentation in:
https://cran.r-project.org/web/packages/schwartz97/vignettes/TechnicalDocument.pdf
'''

'''
Contains functions for pricing under Schwartz97
'''

class Schwartz97:
    def __init__(self, S0, delta0, r, sigma_s, kappa, alpha_tilde, sigma_e, rho):
        self.S0 = S0
        self.delta0 = delta0
        self.r = r
        self.sigma_s = sigma_s
        self.kappa = kappa
        self.alpha_tilde = alpha_tilde
        self.sigma_e = sigma_e
        self.rho = rho

    ##############################################
    ### Path Generator
    ##############################################

    def _Path(self, mu, alpha, timeline, nb_path, W):
        dsigmas = 0.5 * self.sigma_s ** 2
        adjused_drift = (mu - dsigmas)
        T = len(timeline)
        L = zeros([nb_path, T, 2])

        L[:, 0, 0] = log(self.S0) + (adjused_drift - self.delta0) * timeline[0] \
                     + self.sigma_s * sqrt(timeline[0]) * W[:, 0, 0]
        L[:, 0, 1] = self.delta0 + self.kappa * (alpha - self.delta0) * timeline[0] \
                     + self.sigma_e * sqrt(timeline[0]) * W[:, 0, 1]

        for t in range(1, T):
            delta = (timeline[t] - timeline[t - 1])
            sqrtdelta = sqrt(delta)
            L[:, t, 0] = L[:, t - 1, 0] + (adjused_drift - L[:, t - 1, 1]) * delta \
                         + self.sigma_s * sqrtdelta * W[:, t, 0]
            L[:, t, 1] = L[:, t - 1, 1] + self.kappa * (alpha - L[:, t - 1, 1]) * delta \
                         + self.sigma_e * sqrtdelta * W[:, t, 1]
        L[:, :, 0] = exp(L[:, :, 0])
        return L

    def PathP(self, mu, alpha, timeline, nb_path):
        W = multivariate_normal([0, 0], [[1, self.rho], [self.rho, 1]], (nb_path, len(timeline)))
        return self._Path(mu, alpha, timeline, nb_path, W)

    def PathQ(self, timeline, nb_path):
        W = multivariate_normal([0, 0], [[1, self.rho], [self.rho, 1]], (nb_path, len(timeline)))
        return self._Path(self.r, self.alpha_tilde, timeline, nb_path, W)

    '''
    Create consistent path between pricing measure Q and physical measure P
    '''

    def PathPQ(self, mu, alpha, timeline, nb_path):
        W = multivariate_normal([0, 0], [[1, self.rho], [self.rho, 1]], (nb_path, len(timeline)))
        return (self._Path(mu, alpha, timeline, nb_path, W), self._Path(self.r, self.alpha_tilde, timeline, nb_path, W))

    ##############################################
    ### Joint distribution of State Variable
    ##############################################

    def A(self, T):
        cov = self.sigma_s * self.sigma_e * self.rho
        sigma_e_sq = self.sigma_e ** 2
        kappa_sq = self.kappa ** 2
        result = (self.r - self.alpha_tilde + 0.5 * sigma_e_sq / kappa_sq - (cov / self.kappa)) * T \
                 + 0.25 * sigma_e_sq * (1 - exp(-2 * self.kappa * T)) / (self.kappa ** 3) \
                 + (self.kappa * self.alpha_tilde + cov - sigma_e_sq / self.kappa) \
                   * (1 - exp(-self.kappa * T)) / self.kappa ** 2
        return result

    def B(self, T):
        return - (1 - exp(-self.kappa * T)) / self.kappa

    def mu_X(self, S_ini, delta_ini, T):
        return log(S_ini) \
               + (self.r - 0.5 * self.sigma_s ** 2 - self.alpha_tilde) * T \
               + (self.alpha_tilde - delta_ini) * (1 - exp(-self.kappa*T)) / self.kappa

    def mu_delta(self, delta_ini, T):
        return exp(-self.kappa * T) * delta_ini + self.alpha_tilde * (1 - exp(-self.kappa*T))

    def var_X(self, T):
        return (self.sigma_e / self.kappa) ** 2 * \
               ((1 - exp(-2 * self.kappa*T)) / (2 * self.kappa) - 2 / self.kappa * (1 - exp(-self.kappa*T)) + T) \
               + 2 * self.sigma_s * self.sigma_e * self.rho / self.kappa * ((1 - exp(-self.kappa*T)) / self.kappa - T) \
               + self.sigma_s ** 2 * T

    def var_delta(self, T):
        return self.sigma_e ** 2 / (2 * self.kappa) * (1 - exp(-2 * self.kappa * T))

    def cov_Xdelta(self, T):
        return 1 / self.kappa * (
            (self.sigma_s * self.sigma_e * self.rho - self.sigma_e ** 2 / self.kappa) * (1 - exp(-self.kappa * T))
            + self.sigma_e ** 2 / (2 * self.kappa) * (1 - exp(-2 * self.kappa * T)))

    ####################
    ### Financial instrument
    ####################

    def _sigmasq_F(self, t, T):
        return self.sigma_s ** 2 * t \
               + 2 * self.sigma_s * self.sigma_e * self.rho / self.kappa \
                 *(exp(-self.kappa * T) * (exp(self.kappa * t) - 1) / self.kappa - t) \
               + (self.sigma_e / self.kappa) ** 2 \
                 * (t + exp(-2 * self.kappa * T) / (2 * self.kappa) * (exp(2 * self.kappa * t) - 1)
                    - 2 / self.kappa * exp(-self.kappa * T) * (exp(self.kappa * t) - 1))

    def forward(self, t, T, S_ini, delta_ini):
        assert t <= T
        return exp(self.mu_X(S_ini, delta_ini, T-t) + 0.5 * self.var_X(T-t))

    def call(self, t, maturity_option, delivery_time_forward, K, S_ini, delta_ini):

        assert t <= maturity_option
        assert maturity_option <= delivery_time_forward

        mu = self.forward(t, delivery_time_forward, S_ini, delta_ini)
        if t == maturity_option:  # it mean t = maturity_option and there is not stochasticity and no discount factor
            return maximum(mu - K, 0)
        else:
            sigmasq = self._sigmasq_F(maturity_option - t, delivery_time_forward)
            mu = log(mu) - 0.5 * sigmasq
            return generic_call(mu, sigmasq, K) * exp(-self.r * (maturity_option - t))

    def swap(self, t, exchange_time, maturities, S_ini, delta_ini):

        assert len(exchange_time) == len(maturities)
        assert all(less_equal(exchange_time, maturities))
        assert t <= exchange_time[-1]
        # assert both should be sorted

        index_t = bisect_left(exchange_time, t)
        return exp(self.r * t) * sum(
            [exp(-self.r * exchange_time[i]) * self.forward(t, maturities[i], S_ini, delta_ini)
             for i in range(index_t, len(maturities))])

    ####################
    ### getter/setter
    ####################

    def getter(self):
        return self.r, self.sigma_s, self.kappa, self.alpha_tilde, self.sigma_e, self.rho

    def setter(self, mu_s, sigma_s, kappa_delta, mu_delta, sigma_delta, corr):
        self.r = mu_s
        self.sigma_s = sigma_s
        self.kappa = kappa_delta
        self.alpha_tilde = mu_delta
        self.sigma_e = sigma_delta
        self.rho = corr

    ####################
    ### properties
    ####################

    parameters = property(getter, setter)
