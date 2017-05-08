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

    def test_Calibration_hull(self):
        timeline = array([2, 4, 6, 8])  # it crashes when this is not an array
        times_default = [2, 4, 8]
        probability_default = [0.5, 0.2, 0.1]
        b = 0.1
        random.seed(100)
        Z = array([[4, 5.2], [45, 1.6], [45, 12.4], [45, 2.23]])
        a = Calibration_hull(b, Z, timeline, probability_default, times_default)
        p0 = 1 - exp(-a[0] * times_default[0]
                     - b * Z[0, :] * times_default[0])
        p1 = 1 - exp(-a[0] * times_default[0] - a[1] * (times_default[1] - times_default[0])
                     - b * (Z[0, :] * timeline[0] + Z[1, :] * (timeline[1] - timeline[0])))
        p2 = 1 - exp(- a[0] * times_default[0] - a[1] * (times_default[1] - times_default[0])
                     - a[2] * (times_default[2] - times_default[1])
                     - b * (Z[0, :] * timeline[0] + Z[1, :] * (timeline[1] - timeline[0])
                            + Z[2, :] * (timeline[2] - timeline[1]) + Z[3, :] * (timeline[3] - timeline[2])))

        self.assertAlmostEqual(mean(p0), probability_default[0], delta=1E-10)
        self.assertAlmostEqual(mean(p1), probability_default[1], delta=1E-10)
        self.assertAlmostEqual(mean(p2), probability_default[2], delta=1E-10)


if __name__ == '__main__':
    unittest.main()
