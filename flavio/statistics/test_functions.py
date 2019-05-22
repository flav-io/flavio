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

    def test_pull(self):
        self.assertEqual(pull(9, 1), 3)
        self.assertAlmostEqual(pull(2.30, 2), 1, delta=0.006)
        self.assertAlmostEqual(pull(8.03, 3), 2, delta=0.006)
        self.assertAlmostEqual(pull(11.83, 2), 3, delta=0.006)
        self.assertAlmostEqual(pull(delta_chi2(4.52, 8), 8), 4.52, places=10)

    def test_cl(self):
        self.assertAlmostEqual(confidence_level(1), 0.6826894921370859, places=10)
        self.assertAlmostEqual(confidence_level(2), 0.9544997361036416, places=10)
        self.assertAlmostEqual(confidence_level(5), 0.9999994266968562, places=10)

    def test_pvalue(self):
        # trivial
        self.assertAlmostEqual(pvalue(1, 1), 1 - 0.6826894921370859, places=10)
        # non-trivial: http://psychclassics.yorku.ca/Fisher/Methods/tabIII.gif
        self.assertAlmostEqual(pvalue(22.362, 13), 0.05, places=6)
