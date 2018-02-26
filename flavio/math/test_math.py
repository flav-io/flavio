import unittest
import numpy as np
import numpy.testing as npt
import flavio
import scipy.stats

class TestMath(unittest.TestCase):
    def test_normal_pdf(self):
        x = 2.5
        x_arr = np.array([-0.3, 1, 1.5, 2.])
        mu = 0.3
        sigma = 0.92
        # with numbers
        self.assertAlmostEqual(
            flavio.math.functions.normal_logpdf(x, mu, sigma),
            scipy.stats.norm.logpdf(x, mu, sigma), places=10)
        self.assertAlmostEqual(
            flavio.math.functions.normal_pdf(x, mu, sigma),
            scipy.stats.norm.pdf(x, mu, sigma), places=10)
        # with arrays
        npt.assert_array_almost_equal(
            flavio.math.functions.normal_logpdf(x_arr, mu, sigma),
            scipy.stats.norm.logpdf(x_arr, mu, sigma),
            decimal=10)
        npt.assert_array_almost_equal(
            flavio.math.functions.normal_pdf(x_arr, mu, sigma),
            scipy.stats.norm.pdf(x_arr, mu, sigma),
            decimal=10)

    def test_dilog(self):
        # check that mpmath result for complex dilog is reproduced
        self.assertAlmostEqual(flavio.math.functions.li2(1 + 1j),
                         0.6168502750680851+1.4603621167531196j)
        self.assertAlmostEqual(flavio.math.functions.li2(1 - 2j),
                         -0.05947479867380914-2.072647971774756j)
        self.assertAlmostEqual(flavio.math.functions.li2(1),
                         1.6449340668482264)
        self.assertAlmostEqual(flavio.math.functions.li2(-1),
                         -0.8224670334241132)
