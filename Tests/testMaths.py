import unittest

from numpy import log, sqrt, exp, maximum, mean
from numpy.random import multivariate_normal, seed

import Maths.ClosedForm.BlackScholes as bls
import Maths.ClosedForm.GenericFormulas as formulas


class testMaths(unittest.TestCase):

    def test_BlackScholes(self):
        S0 = 100
        r = 0.1
        div = 0.05
        sigma = 0.2
        K = 100
        T = 1

        self.assertAlmostEqual(bls.Call(S0=S0, sigma=sigma, r=r, K=K, T=T, div=div), 9.940902597066710,
                               delta=1E-10)
        self.assertAlmostEqual(bls.Put(S0=S0, sigma=sigma, r=r, K=K, T=T, div=div), 5.301701950591252,
                               delta=1E-10)
        self.assertAlmostEqual(bls.Call_Delta(S0=S0, sigma=sigma, r=r, K=K, T=T, div=div), 0.605772053822199,
                               delta=1E-10)
        self.assertAlmostEqual(bls.Call_Gamma(S0=S0, sigma=sigma, r=r, K=K, T=T, div=div), 0.017846982962362,
                               delta=1E-10)
        self.assertAlmostEqual(bls.Call_Vega(S0=S0, sigma=sigma, r=r, K=K, T=T, div=div), 35.693965924724715,
                               delta=1E-10)
        self.assertAlmostEqual(bls.Call_Theta(S0=S0, sigma=sigma, r=r, K=K, T=T, div=div), -5.604166601876797,
                               delta=1E-10)
        self.assertAlmostEqual(bls.Call_Rho(S0=S0, sigma=sigma, r=r, K=K, T=T, div=div), 50.636302785153205,
                               delta=1E-10)

    def test_GenericFormula(self):
        S0 = 100
        r = 0.1
        div = 0.05
        sigma = 0.5
        K = 100
        T = 1

        mu = log(S0) + (r - div - 0.5*sigma**2)*T
        variance_tot = (sigma**2)*T
        self.assertAlmostEqual(bls.Call(S0=S0, sigma=sigma, r=r, K=K, T=T, div=div)*exp(r*T),
                               formulas.Call(mu=mu, sigmasq=variance_tot, K=K),
                               delta=1E-10)


if __name__ == '__main__':
    unittest.main()
