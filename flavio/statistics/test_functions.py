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
