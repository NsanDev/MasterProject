from Scripts.CreditModel.Tools.RiskStatistics import risk_statistics
from numpy import ones

import unittest

class TestStringMethods(unittest.TestCase):

    def test_risk_statistics(self):
        A = range(100,0,-1)
        W = ones(100)
        W = W / 100
        results1 = risk_statistics(A, W, 0.1)
        results2 = risk_statistics(A, alpha=0.1)
        self.assertEqual(results1, [50.5, 10, 90, 95])
        self.assertEqual(results2, [50.5, 10, 90, 95])

if __name__ == '__main__':
    unittest.main()