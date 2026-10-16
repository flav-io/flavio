import unittest
import numpy as np
import flavio
import scipy.stats
from flavio.statistics.functions import *

class TestFunctions(unittest.TestCase):
    def test_deltachi2(self):
        self.assertEqual(delta_chi2(3, 1), 9)
        self.assertAlmostEqual(delta_chi2(1, 2), 2.30, delta=0.006)
        self.assertAlmostEqual(delta_chi2(2, 3), 8.03, delta=0.006)
        self.assertAlmostEqual(delta_chi2(3, 2), 11.83, delta=0.006)

    def test_large_nsigma(self):
        # no saturation for large nsigma
        d = delta_chi2(9, 2)
        self.assertTrue(np.isfinite(d))
        self.assertAlmostEqual(d, 2*np.log(1/(2*scipy.stats.norm.sf(9))), delta=1)
        self.assertAlmostEqual(pull(d, 2) / 9, 1, places=6)
        self.assertTrue(pvalue(200, 2) > 0)
        self.assertAlmostEqual(pvalue(200, 2) / np.exp(-100), 1, places=10)

    def test_agree_unsaturated(self):
        for n in [0.1, 0.5, 1, 2, 3, 4, 5]:
            cl = (scipy.stats.norm.cdf(n) - 0.5) * 2
            self.assertAlmostEqual(confidence_level(n) / cl, 1, places=12)
            for dof in [2, 3, 8]:
                old = scipy.stats.chi2(dof).ppf(cl)
                self.assertAlmostEqual(delta_chi2(n, dof) / old, 1, places=10)
        for x in [0.5, 3, 20]:
            for dof in [1, 2, 5]:
                old = 1 - scipy.stats.chi2.cdf(x, dof)
                self.assertAlmostEqual(pvalue(x, dof) / old, 1, places=10)

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
