import unittest

from numpy import array, ones, less_equal, transpose, random

from CreditModel.DirectionalWayRisk.Weights import *
from CreditModel.DirectionalWayRisk.Weights import _integrate_intensity
from CreditModel.Tools.RiskStatistics import risk_statistics


class testCreditModel(unittest.TestCase):
    def test_integral_intensity(self):
        h = array([2, 4, 6])
        times = array([10, 20, 30])
        integral_intensity = lambda t_up: _integrate_intensity(t_up, h_rates=h, timeline=times)
        self.assertEqual(integral_intensity(0.5), 0.5 * 2)
        self.assertEqual(integral_intensity(5), 5 * 2)
        self.assertEqual(integral_intensity(10), 10 * 2)
        self.assertEqual(integral_intensity(15), 10 * 2 + (15 - 10) * 4)
        self.assertEqual(integral_intensity(20), 10 * 2 + (20 - 10) * 4)
        self.assertEqual(integral_intensity(25), 10 * 2 + (20 - 10) * 4 + (25 - 20) * 6)
        self.assertEqual(integral_intensity(30), 10 * 2 + (20 - 10) * 4 + (30 - 20) * 6)

    def test_risk_statistics(self):
        A = range(100, 0, -1)
        W = ones(100)
        W = W / 100
        results1 = risk_statistics(A, W, 0.1)
        results2 = risk_statistics(A, alpha=0.1)
        self.assertEqual(results1, [50.5, 10, 90, 95])
        self.assertEqual(results2, [50.5, 10, 90, 95])

    def test_Merton(self):
        Z_M = 50 * ones(100)
        weights = Merton(Z_M, rho=0, probability_default=0.5, tolerance=0.001)
        [self.assertAlmostEqual(weights[k], 1.0 / len(Z_M)) for k in range(0, len(Z_M))]
        self.assertTrue(all(less_equal(weights, 1)))
        self.assertAlmostEqual(sum(weights), 1)

        weights = Merton(Z_M, rho=0, probability_default=0.1, tolerance=0.001)
        [self.assertAlmostEqual(weights[k], 1.0 / len(Z_M)) for k in range(0, len(Z_M))]
        self.assertTrue(all(less_equal(weights, 1)))
        self.assertAlmostEqual(sum(weights), 1)

        weights = Merton(Z_M, rho=0.5, probability_default=0.1, tolerance=0.001)
        self.assertTrue(all(less_equal(weights, 1)))
        self.assertAlmostEqual(sum(weights), 1)

        weights = Merton(Z_M, rho=-0.5, probability_default=0.1, tolerance=0.001)
        self.assertTrue(all(less_equal(weights, 1)))
        self.assertAlmostEqual(sum(weights), 1)

    def test_Weights(self):
        '''
        Test if the weights are correctly calculated with constant hazard rate
        '''
        timeline = array([2, 4, 6, 8, 10])  # it crashes when this is not an array
        time_exposure = [4, 6, 8]
        l = [ones(len(timeline)) / 10, ones(len(timeline)) / 5]
        hazard_rates = transpose(l)  # array(list(map(array, zip(*l))))
        weights = Weights(hazard_rates=hazard_rates, timeline=timeline, times_exposure=time_exposure)
        PD1 = lambda t1, t2: exp(-t1 / 10) - exp(-t2 / 10)
        PD2 = lambda t1, t2: exp(-t1 / 5) - exp(-t2 / 5)
        actual_weights = [[PD1(0, 4) / (PD1(0, 4) + PD2(0, 4)), PD2(0, 4) / (PD1(0, 4) + PD2(0, 4))],
                          [PD1(4, 6) / (PD1(4, 6) + PD2(4, 6)), PD2(4, 6) / (PD1(4, 6) + PD2(4, 6))],
                          [PD1(6, 8) / (PD1(6, 8) + PD2(6, 8)), PD2(6, 8) / (PD1(6, 8) + PD2(6, 8))]]

        for i in range(0, len(time_exposure)):
            for j in range(0, len(weights[0])):
                self.assertAlmostEqual(actual_weights[i][j], weights[i][j])

        '''
        Test if the weights are correctly calculated with time varying hazard rate
        '''
        timeline = array([2, 4, 6, 8])  # it crashes when this is not an array
        time_exposure = [4, 8]
        l = [[0.1, 0.3, 0.2, 0.5], [0.2, 0.6, 0.4, 0.25]]
        hazard_rates = transpose(l)  # array(list(map(array, zip(*l))))
        weights = Weights(hazard_rates=hazard_rates, timeline=timeline, times_exposure=time_exposure)
        PD = lambda h, t1, t2: exp(-h * t1) - exp(-h * t2)

        pd11 = 1 - exp(-0.1 * 2 - (4 - 2) * 0.3)
        pd12 = 1 - exp(-0.2 * 2 - (4 - 2) * 0.6)
        pd21 = exp(-0.1 * 2 - (4 - 2) * 0.3) \
               - exp(-0.1 * 2 - (4 - 2) * 0.3 - (6 - 4) * 0.2 - (8 - 6) * 0.5)
        pd22 = exp(-0.2 * 2 - (4 - 2) * 0.6) \
               - exp(-0.2 * 2 - (4 - 2) * 0.6 - (6 - 4) * 0.4 - (8 - 6) * 0.25)
        actual_weights = [[pd11 / (pd11 + pd12), pd12 / (pd11 + pd12)], [pd21 / (pd21 + pd22), pd22 / (pd21 + pd22)]]
        for i in range(0, len(time_exposure)):
            for j in range(0, len(weights[0])):
                self.assertAlmostEqual(actual_weights[i][j], weights[i][j])

    def test_hull(self):
        timeline = array([2, 4, 6, 6.5])  # it crashes when this is not an array
        times_default = [2, 4, 6.5]
        times_exposure = [4, 6.5]

        survival_probability = [0.5, 0.2, 0.1]
        b = 0.001
        random.seed(100)
        Z = array([[4, 5.2], [4.5, 5.6], [4.1, 5.4], [4.2, 5.23]])

        max_iter = 100000
        tol = 1e-7

        Z_M = b * Z
        a = Calibration_hull(Z_M, timeline, survival_probability, times_default, max_iter, tol)

        h = a + Z_M
        h = exp(h)
        p0 = exp(-h[0] * (timeline[0]))
        p1 = exp(-h[0] * (timeline[0]) - h[1] * (timeline[1] - timeline[0]))
        p2 = exp(-h[0] * (timeline[0]) - h[1] * (timeline[1] - timeline[0]) - h[2] * (timeline[2] - timeline[1])
                 - h[3] * (timeline[3] - timeline[2]))

        z = mean(exp(-_integrate_intensity(times_default[2], exp(a + b * Z), timeline))) - survival_probability[2]

        self.assertAlmostEqual(mean(p0), survival_probability[0], delta=tol)
        self.assertAlmostEqual(mean(p1), survival_probability[1], delta=tol)
        self.assertAlmostEqual(mean(p2), survival_probability[2], delta=tol)

        calculated_weights = Hull(Z_M, timeline, survival_probability, times_default, times_exposure
                                  , max_iter=max_iter, tol=tol)
        w1 = (1 - p1) / sum(1 - p1)
        w2 = (p2 - p1) / sum(p2 - p1)
        expected_weights = array([w1, w2])
        # self.assertAlmostEqual(w1[0], calculated_weights[0, 0])
        self.assertAlmostEqual(w1[1], calculated_weights[0, 1])
        self.assertAlmostEqual(w2[0], calculated_weights[1, 0])
        self.assertAlmostEqual(w2[1], calculated_weights[1, 1])


if __name__ == '__main__':
    unittest.main()
