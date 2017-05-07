import unittest

from numpy import array, ones

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


if __name__ == '__main__':
    unittest.main()
