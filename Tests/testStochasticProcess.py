'''
Created on 21 mars 2017

@author: Naitra
'''
import unittest

from numpy import linspace
from numpy import random, array_equal, mean
from scipy.stats import norm

from Maths.ClosedForm.BlackScholes import Call
from StochasticProcess.Commodities.Schwartz97 import *
from StochasticProcess.GeometricBrownianMotion import GeometricBrownianMotion as GBM


class testStochasticProcess(unittest.TestCase):

    def test_GeometricBrownianMotion(self):
        S0 = 100
        drift = 0.1
        vol = 0.1
        T = 1
        random.seed(600)

        gbm = GBM(S0,drift,vol)
        timeline = linspace(0, T, 100, endpoint=True)
        nbSimulations = 100000
        paths = gbm.Path(timeline, nbSimulations)
        K = 95
        r = 0.1
        actual_value_call = Call(S0, r, vol, K, T)
        estimated_value_call = mean(maximum(paths[:, -1]-K, 0)*exp(-r*T))
        self.assertAlmostEqual(actual_value_call, estimated_value_call, delta=1E-2)

    def test_Schwartz97(self):
        S0 = 45
        delta0 = 0.1
        r = 0.142
        sigma_s = 0.393
        kappa = 1.876
        alpha = 0.393
        sigma_e = 0.527
        rho = 0.766

        model = Schwartz97(S0=S0, delta0=delta0, r=r, sigma_s=sigma_s, kappa=kappa, alpha_tilde=alpha, sigma_e=sigma_e,
                           rho=rho)


        ########################################
        ### joint distribution log(S), delta
        ########################################

        self.assertEqual(model.A(0), 0)
        self.assertEqual(model.B(0), 0)
        self.assertEqual(model.mu_X(S0, delta0, 0), log(S0))
        self.assertEqual(model.var_X(0), 0)
        self.assertEqual(model.var_delta(0), 0)
        self.assertEqual(model.cov_Xdelta(0), 0)


        ########################################
        ### Forward
        ########################################

        self.assertAlmostEqual(model.forward(0, 0, S0, delta0), S0, delta=1E-10)
        self.assertEqual(model.forward(0, 2, S0, delta0), S0 * exp(model.A(2) + model.B(2) * delta0))
        self.assertAlmostEqual(model.forward(0, 0.75, S0, delta0), S0 * exp(model.A(0.75) + model.B(0.75) * delta0)
                               , delta=1E-10)
        self.assertAlmostEqual(model.forward(0, 5.654, S0, delta0), S0 * exp(model.A(5.654) + model.B(5.654) * delta0)
                         , delta=1E-10)
        self.assertEqual(model.forward(0, 5.654, S0, delta0),
                         model.swap(t=0, exchange_time=[0], maturities=[5.654], S_ini=S0, delta_ini=delta0))


        ########################################
        ### Swap
        ########################################

        t = 0.56977
        self.assertAlmostEqual(model.forward(t, 1.1, S0, delta0)*exp(-r*(1 - t))
                         + model.forward(t, 1.6, S0, delta0)*exp(-r*(1.5 - t))
                         + model.forward(t, 1.8, S0, delta0) * exp(-r*(1.75 - t)),
                         model.swap(t=t, exchange_time=[1,1.5,1.75], maturities=[1.1,1.6,1.8], S_ini=S0,
                                    delta_ini=delta0))
        t = 1
        self.assertAlmostEqual(model.forward(t, 1.1, S0, delta0)*exp(-r*(1 - t))
                         + model.forward(t, 1.6, S0, delta0)*exp(-r*(1.5 - t))
                         + model.forward(t, 1.8, S0, delta0) * exp(-r*(1.75 - t)),
                         model.swap(t=t, exchange_time=[1, 1.5, 1.75], maturities=[1.1, 1.6, 1.8], S_ini=S0,
                                    delta_ini=delta0))
        t = 1.2
        self.assertAlmostEqual(model.forward(t, 1.6, S0, delta0)*exp(-r*(1.5 - t))
                         + model.forward(t, 1.8, S0, delta0) * exp(-r*(1.75 - t)),
                         model.swap(t=t, exchange_time=[1, 1.5, 1.75], maturities=[1.1, 1.6, 1.8], S_ini=S0,
                                    delta_ini=delta0))
        t = 1.4446
        self.assertAlmostEqual(model.forward(t, 1.6, S0, delta0)*exp(-r*(1.5 - t))
                         + model.forward(t, 1.8, S0, delta0) * exp(-r*(1.75 - t)),
                         model.swap(t=t, exchange_time=[1, 1.5, 1.75], maturities=[1.1, 1.6, 1.8], S_ini=S0,
                                    delta_ini=delta0))

        t = 1.5
        self.assertAlmostEqual(model.forward(t, 1.6, S0, delta0)*exp(-r*(1.5 - t))
                         + model.forward(t, 1.8, S0, delta0) * exp(-r*(1.75 - t)),
                         model.swap(t=t, exchange_time=[1, 1.5, 1.75], maturities=[1.1, 1.6, 1.8], S_ini=S0,
                                    delta_ini=delta0))
        t = 1.6
        self.assertAlmostEqual(model.forward(t, 1.8, S0, delta0) * exp(-r*(1.75 - t)),
                         model.swap(t=t, exchange_time=[1, 1.5, 1.75], maturities=[1.1, 1.6, 1.8], S_ini=S0,
                                    delta_ini=delta0))
        t = 1.714385
        self.assertAlmostEqual(model.forward(t, 1.8, S0, delta0) * exp(-r*(1.75 - t)),
                         model.swap(t=t, exchange_time=[1, 1.5, 1.75], maturities=[1.1, 1.6, 1.8], S_ini=S0,
                                    delta_ini=delta0))
        t = 1.75
        self.assertAlmostEqual(model.forward(t, 1.8, S0, delta0) * exp(-r*(1.75 - t)),
                         model.swap(t=t, exchange_time=[1, 1.5, 1.75], maturities=[1.1, 1.6, 1.8], S_ini=S0,
                                    delta_ini=delta0))


        ########################################
        ### Call and Put
        ########################################

        t, T, K, TM = 0, 0.75, 30, 1.6

        ### Closed Form valuation ###
        G = model.forward(t, TM, S0, delta0)
        sigmasq = model._sigmasq_F(T-t, TM)
        dp, dm = (log(G/K) + 0.5*sigmasq)/sqrt(sigmasq), (log(G/K) - 0.5*sigmasq)/sqrt(sigmasq)
        actual_value = exp(-r*(T-t))*(G*norm.cdf(dp)-K*norm.cdf(dm))
        calculated_call = model.call(t=t, maturity_option=T, delivery_time_forward=TM, K=K, S_ini=S0, delta_ini=delta0)
        self.assertAlmostEqual(actual_value, calculated_call, delta=1E-10)

        calculated_put = model.put(t=t, maturity_option=T, delivery_time_forward=TM, K=K, S_ini=S0, delta_ini=delta0)

        self.assertAlmostEqual(calculated_call - calculated_put,
                               exp(-r * (T - t)) * (model.forward(t, TM, S0, delta0) - K))

        ### MC valuation ###
        random.seed(600)
        timeline = linspace(0.01, T - t, num=20, endpoint=True)
        pathQ = model.PathQ(timeline=timeline, nb_path=100000)
        ForwardtT = pathQ[:, -1, 0]*exp(model.A(TM-T) + model.B(TM-T)*pathQ[:, -1, 1])
        estimate_call = maximum(ForwardtT-K, 0) * exp(-r*(T-t))
        self.assertAlmostEqual(actual_value, mean(estimate_call), delta=1E-2)


        ########################################
        ### MC Generator
        ########################################

        random.seed(600)
        pathPQ = model.PathPQ(r, alpha, timeline=timeline, nb_path=10)
        random.seed(600)
        pathQ = model.PathQ(timeline=timeline, nb_path=10)
        random.seed(600)
        pathP = model.PathP(r, alpha, timeline=timeline, nb_path=10)

        assert array_equal(pathPQ[0], pathPQ[1]) # test if pathP and pathQ gives the same results
        assert array_equal(pathPQ[0], pathP)
        assert array_equal(pathPQ[1], pathQ)


if __name__ == "__main__":
    unittest.main()