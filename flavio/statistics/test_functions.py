import unittest
import numpy as np
import flavio
from flavio.statistics.functions import *

class TestFunctions(unittest.TestCase):
    def test_deltachi2(self):
        self.assertEqual(delta_chi2(3, 1), 9)
        self.assertAlmostEqual(delta_chi2(1, 2), 2.30, delta=0.006)
        self.assertAlmostEqual(delta_chi2(2, 3), 8.03, delta=0.006)
        self.assertAlmostEqual(delta_chi2(3, 2), 11.83, delta=0.006)

    def test_cl(self):
        self.assertAlmostEqual(confidence_level(1), 0.6826894921370859, places=10)
        self.assertAlmostEqual(confidence_level(2), 0.9544997361036416, places=10)
        self.assertAlmostEqual(confidence_level(5), 0.9999994266968562, places=10)
