import unittest
import numpy as np
import numpy.testing as npt
import flavio
import scipy.stats
from math import pi, sqrt, exp, log
from flavio.statistics.probability import *

class TestProbability(unittest.TestCase):
    def test_multiv_normal(self):
        # test that the rescaling of the MultivariateNormalDistribution
        # does not affect the log PDF!
        c = np.array([1e-3, 2])
        cov = np.array([[(0.2e-3)**2, 0.2e-3*0.5*0.3],[0.2e-3*0.5*0.3, 0.5**2]])
        pdf = MultivariateNormalDistribution(c, cov)
        x=np.array([1.5e-3, 0.8])
        num_lpdf = pdf.logpdf(x)
        ana_lpdf = log(1/sqrt(4*pi**2*np.linalg.det(cov))*exp(-np.dot(np.dot(x-c,np.linalg.inv(cov)),x-c)/2))
        self.assertAlmostEqual(num_lpdf, ana_lpdf, delta=1e-6)

    def test_halfnormal(self):
        pdf_p_1 = HalfNormalDistribution(1.7, 0.3)
        pdf_n_1 = HalfNormalDistribution(1.7, -0.3)
        pdf_p_2 = AsymmetricNormalDistribution(1.7, 0.3, 0.0001)
        pdf_n_2 = AsymmetricNormalDistribution(1.7, 0.0001, 0.3)
        self.assertAlmostEqual(pdf_p_1.logpdf(1.99), pdf_p_2.logpdf(1.99), delta=0.001)
        self.assertEqual(pdf_p_1.logpdf(1.55), -np.inf)
        self.assertAlmostEqual(pdf_n_1.logpdf(1.55), pdf_n_2.logpdf(1.55), delta=0.001)
        self.assertEqual(pdf_n_1.logpdf(1.99), -np.inf)

    def test_limit(self):
        p1 = GaussianUpperLimit(1.78, 0.68268949)
        p2 = HalfNormalDistribution(0, 1.78)
        self.assertAlmostEqual(p1.logpdf(0.237), p2.logpdf(0.237), delta=0.0001)
        self.assertEqual(p2.logpdf(-1), -np.inf)

    def test_numerical(self):
        x = np.arange(-5,7,0.01)
        y = scipy.stats.norm.pdf(x, loc=1)
        y_crazy = 14.7 * y # multiply PDF by crazy number
        p_num = NumericalDistribution(x, y_crazy)
        p_norm = NormalDistribution(1, 1)
        self.assertAlmostEqual(p_num.logpdf(0.237), p_norm.logpdf(0.237), delta=0.02)
        self.assertAlmostEqual(p_num.logpdf(-2.61), p_norm.logpdf(-2.61), delta=0.02)
        self.assertAlmostEqual(p_num.ppf_interp(0.1), scipy.stats.norm.ppf(0.1, loc=1), delta=0.02)
        self.assertAlmostEqual(p_num.ppf_interp(0.95), scipy.stats.norm.ppf(0.95, loc=1), delta=0.02)
        # just check if this raises an error
        p_num.get_random(100)

    def test_numerical_from_analytic(self):
        p_norm = NormalDistribution(1.64, 0.32)
        p_norm_num = NumericalDistribution.from_pd(p_norm)
        self.assertEqual(p_norm.central_value, p_norm_num.central_value)
        self.assertEqual(p_norm.support, p_norm_num.support)
        npt.assert_array_almost_equal(p_norm.logpdf([0.7, 1.9]), p_norm_num.logpdf([0.7, 1.9]), decimal=3)
        p_asym = AsymmetricNormalDistribution(1.64, 0.32, 0.67)
        p_asym_num = NumericalDistribution.from_pd(p_asym)
        npt.assert_array_almost_equal(p_asym.logpdf([0.7, 1.9]), p_asym_num.logpdf([0.7, 1.9]), decimal=3)
        p_unif = UniformDistribution(1.64, 0.32)
        p_unif_num = NumericalDistribution.from_pd(p_unif)
        npt.assert_array_almost_equal(p_unif.logpdf([0.7, 1.9]), p_unif_num.logpdf([0.7, 1.9]), decimal=3)
        p_half = HalfNormalDistribution(1.64, -0.32)
        p_half_num = NumericalDistribution.from_pd(p_half)
        npt.assert_array_almost_equal(p_half.logpdf([0.7, 1.3]), p_half_num.logpdf([0.7, 1.3]), decimal=3)

    def test_convolve_normal(self):
        p_1 = NormalDistribution(12.4, 0.346)
        p_2 = NormalDistribution(12.4, 2.463)
        p_x = NormalDistribution(12.3, 2.463)
        from flavio.statistics.probability import convolve_distributions
        # error if not the same central value:
        with self.assertRaises(AssertionError):
            convolve_distributions([p_1, p_x])
        p_comb = convolve_distributions([p_1, p_2])
        self.assertIsInstance(p_comb, NormalDistribution)
        self.assertEqual(p_comb.central_value, 12.4)
        self.assertEqual(p_comb.standard_deviation, sqrt(0.346**2+2.463**2))

    def test_convolve_numerical(self):
        from flavio.statistics.probability import _convolve_numerical
        p_1 = NumericalDistribution.from_pd(NormalDistribution(12.4, 0.346))
        p_2 = NumericalDistribution.from_pd(NormalDistribution(12.4, 2.463))
        p_3 = NumericalDistribution.from_pd(NormalDistribution(12.4, 1.397))
        conv_p_12 = _convolve_numerical([p_1, p_2])
        comb_p_12 = NormalDistribution(12.4, sqrt(0.346**2 + 2.463**2))
        conv_p_123 = _convolve_numerical([p_1, p_2, p_3])
        comb_p_123 = NormalDistribution(12.4, sqrt(0.346**2 + 2.463**2 + 1.397**2))
        x = np.linspace(2, 20, 10)
        npt.assert_array_almost_equal(conv_p_12.logpdf(x), comb_p_12.logpdf(x), decimal=1)
        npt.assert_array_almost_equal(conv_p_123.logpdf(x), comb_p_123.logpdf(x), decimal=1)
