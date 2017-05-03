import unittest
from numpy import array

from CreditModel.DirectionalWayRisk.Weights import _integrate_intensity

class WeightsTests(unittest.TestCase):

    def test_risk_statistics(self):
        h = array([2,4,6])
        times = array([10,20,30])
        integral_intensity = lambda t_up: _integrate_intensity(t_up, h_rates=h, timeline=times)
        self.assertEqual(integral_intensity(0.5), 0.5*2)
        self.assertEqual(integral_intensity(5), 5 * 2)
        self.assertEqual(integral_intensity(10), 10 * 2)
        self.assertEqual(integral_intensity(15), 10*2 + (15-10)*4)
        self.assertEqual(integral_intensity(20), 10 * 2 + (20 - 10) * 4)
        self.assertEqual(integral_intensity(25), 10 * 2 + (20 - 10) * 4 + (25-20)*6)
        self.assertEqual(integral_intensity(30), 10 * 2 + (20 - 10) * 4 + (30 - 20) * 6)


if __name__ == '__main__':
    unittest.main()