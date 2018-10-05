import unittest
import numpy as np
import numpy.testing as npt
import scipy.stats
from math import pi, sqrt, exp, log
from flavio.statistics.probability import *
import itertools
import yaml

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
        self.assertEqual(len(pdf.get_random(10)), 10)

    def test_normal(self):
        d = NormalDistribution(2, 0.3)
        self.assertEqual(d.cdf(2), 0.5)
        self.assertEqual(d.ppf(0.5), 2)

    def test_halfnormal(self):
        pdf_p_1 = HalfNormalDistribution(1.7, 0.3)
        pdf_n_1 = HalfNormalDistribution(1.7, -0.3)
        pdf_p_2 = AsymmetricNormalDistribution(1.7, 0.3, 0.0001)
        pdf_n_2 = AsymmetricNormalDistribution(1.7, 0.0001, 0.3)
        self.assertAlmostEqual(pdf_p_1.logpdf(1.99), pdf_p_2.logpdf(1.99), delta=0.001)
        self.assertEqual(pdf_p_1.logpdf(1.55), -np.inf)
        self.assertAlmostEqual(pdf_n_1.logpdf(1.55), pdf_n_2.logpdf(1.55), delta=0.001)
        self.assertEqual(pdf_n_1.logpdf(1.99), -np.inf)
        self.assertEqual(len(pdf_p_1.get_random(10)), 10)
        self.assertEqual(len(pdf_p_2.get_random(10)), 10)
        d = HalfNormalDistribution(2, 0.3)
        self.assertEqual(d.cdf(2), 0.0)
        self.assertAlmostEqual(d.cdf(2.3), 0.6827, places=4)
        self.assertAlmostEqual(d.ppf(0.6827), 2.3, places=4)


    def test_lognormal(self):
        with self.assertRaises(ValueError):
            LogNormalDistribution(1, 0.8)
        with self.assertRaises(ValueError):
            LogNormalDistribution(1, -1.2)
        pdf = LogNormalDistribution(3, 2)
        self.assertAlmostEqual(pdf.get_error_left(), 1.5)
        self.assertAlmostEqual(pdf.get_error_right(), 3)
        pdf2 = LogNormalDistribution(-3, 2)
        self.assertAlmostEqual(pdf2.get_error_right(), 1.5)
        self.assertAlmostEqual(pdf2.get_error_left(), 3)
        self.assertEqual(pdf2.pdf(-2.7), pdf.pdf(2.7))
        self.assertEqual(pdf2.cdf(-2.7), 1 - pdf.cdf(2.7))
        self.assertEqual(pdf2.ppf(0.25), -pdf.ppf(0.75))

    def test_limit(self):
        p1 = GaussianUpperLimit(2*1.78, 0.9544997)
        p2 = HalfNormalDistribution(0, 1.78)
        self.assertAlmostEqual(p1.logpdf(0.237), p2.logpdf(0.237), delta=0.0001)
        self.assertEqual(p2.logpdf(-1), -np.inf)
        self.assertAlmostEqual(p1.cdf(2*1.78), 0.9544997, delta=0.0001)

    def test_gamma(self):
        # check for loc above and below a-1
        for  loc in (-5, -15):
            p = GammaDistribution(a=11, loc=loc, scale=1)
            self.assertEqual(p.central_value, loc + 10)
            r = p.get_random(10)
            self.assertEqual(len(r), 10)
            self.assertAlmostEqual(p.cdf(p.support[1]), 1-2e-9, delta=0.1e-9)
            self.assertAlmostEqual(p.ppf(1-2e-9), p.support[1], delta=0.0001)
            self.assertEqual(loc, p.support[0])
        # nearly normal distribution
        p = GammaDistribution(a=10001, loc=0, scale=1)
        self.assertAlmostEqual(p.error_left, sqrt(10000), delta=1)
        self.assertAlmostEqual(p.get_error_left(nsigma=2), 2*sqrt(10000), delta=2)
        self.assertAlmostEqual(p.error_right, sqrt(10000), delta=1)
        self.assertAlmostEqual(p.get_error_right(nsigma=2), 2*sqrt(10000), delta=2)

    def test_gamma_positive(self):
        # check for loc above and below a-1
        for  loc in (-5, -15):
            p = GammaDistributionPositive(a=11, loc=loc, scale=1)
            self.assertEqual(p.central_value, max(loc + 10, 0))
            r = p.get_random(10)
            self.assertEqual(len(r), 10)
            self.assertTrue(np.min(r) >= 0)
            self.assertEqual(p.logpdf(-0.1), -np.inf)
            self.assertEqual(p.cdf(0), 0)
            self.assertAlmostEqual(p.cdf(p.support[1]), 1-2e-9, delta=0.1e-9)
            self.assertAlmostEqual(p.ppf(0), 0, places=14)
            self.assertAlmostEqual(p.ppf(1-2e-9), p.support[1], delta=0.0001)
            self.assertEqual(p.cdf(-1), 0)
        p = GammaDistributionPositive(a=11, loc=-9, scale=1)
        self.assertEqual(p.central_value, 1)
        self.assertEqual(p.error_left, 1)
        # nearly normal distribution
        p = GammaDistributionPositive(a=10001, loc=0, scale=1)
        self.assertAlmostEqual(p.error_left, sqrt(10000), delta=1)
        self.assertAlmostEqual(p.get_error_left(nsigma=2), 2*sqrt(10000), delta=2)
        self.assertAlmostEqual(p.error_right, sqrt(10000), delta=1)
        self.assertAlmostEqual(p.get_error_right(nsigma=2), 2*sqrt(10000), delta=2)

    def test_gamma_limit(self):
        p = GammaUpperLimit(counts_total=30, counts_background=10,
                            limit=2e-5, confidence_level=0.68)
        self.assertAlmostEqual(p.cdf(2e-5), 0.68, delta=0.0001)
        # no counts
        p = GammaUpperLimit(counts_total=0, counts_background=0,
                            limit=2e-5, confidence_level=0.68)
        self.assertAlmostEqual(p.cdf(2e-5), 0.68, delta=0.0001)
        # background excess
        p = GammaUpperLimit(counts_total=30, counts_background=50,
                            limit=2e5, confidence_level=0.68)
        self.assertAlmostEqual(p.cdf(2e5), 0.68, delta=0.0001)
        p = GammaUpperLimit(counts_total=10000, counts_background=10000,
                            limit=3., confidence_level=0.95)
        p_norm = GaussianUpperLimit(limit=3., confidence_level=0.95)
        # check that large-statistics Gamma and Gauss give nearly same PDF
        for x in [0, 1, 2, 3, 4]:
            self.assertAlmostEqual(p.logpdf(x), p_norm.logpdf(x), delta=0.1)

    def test_general_gamma_limit(self):
        p = GeneralGammaUpperLimit(counts_total=30, counts_background=10,
                            limit=2e-5, confidence_level=0.68,
                            background_variance=5)
        self.assertAlmostEqual(p.cdf(2e-5), 0.68, delta=0.0001)
        # background excess
        p = GeneralGammaUpperLimit(counts_total=30, counts_background=50,
                            limit=2e5, confidence_level=0.68,
                            background_variance=25)
        self.assertAlmostEqual(p.cdf(2e5), 0.68, delta=0.0001)
        p = GeneralGammaUpperLimit(counts_total=10000, counts_background=10000,
                            limit=3., confidence_level=0.95,
                            background_variance=1000)
        p_norm = GaussianUpperLimit(limit=3., confidence_level=0.95)
        # check that large-statistics Gamma and Gauss give nearly same PDF
        for x in [1, 2, 3, 4]:
            self.assertAlmostEqual(p.logpdf(x), p_norm.logpdf(x), delta=0.1)
        # check that warning is raised for very small background variance
        with self.assertWarns(Warning):
            GeneralGammaUpperLimit(counts_total=10000, counts_background=10000,
                            limit=3., confidence_level=0.95,
                            background_variance=10)


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
        self.assertEqual(len(p_num.get_random(10)), 10)

    def test_multiv_numerical(self):
        x0 = np.arange(-5,5,0.01)
        x1 = np.arange(-4,6,0.02)
        cov = [[0.2**2, 0.5*0.2*0.4], [0.5*0.2*0.4, 0.4**2]]
        y = scipy.stats.multivariate_normal.pdf(np.array(list(itertools.product(x0, x1))), mean=[0, 1], cov=cov)
        y = y.reshape(len(x0), len(x1))
        y_crazy = 14.7 * y # multiply PDF by crazy number
        p_num = MultivariateNumericalDistribution((x0, x1), y_crazy)
        p_norm = MultivariateNormalDistribution([0, 1], cov)
        self.assertAlmostEqual(p_num.logpdf([0.237, 0.346]), p_norm.logpdf([0.237, 0.346]), delta=0.02)
        self.assertAlmostEqual(p_num.logpdf([0.237], exclude=(1,)),
                               p_norm.logpdf([0.237], exclude=(1,)), delta=0.02)
        # try again with length-2 xi
        p_num = MultivariateNumericalDistribution(([-5, 4.99], [-4, 5.98]), y_crazy)
        self.assertAlmostEqual(p_num.logpdf([0.237, 0.346]), p_norm.logpdf([0.237, 0.346]), delta=0.02)
        self.assertAlmostEqual(p_num.logpdf([0.237], exclude=(1,)),
                               p_norm.logpdf([0.237], exclude=(1,)), delta=0.02)
        # test exceptions
        with self.assertRaises(NotImplementedError):
            p_num.error_left
        with self.assertRaises(NotImplementedError):
            p_num.error_right
        self.assertEqual(len(p_num.get_random(10)), 10)

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

    def test_numerical_from_analytic_mv(self):
        p = MultivariateNormalDistribution([2, 5], [[(0.2)**2, 0.2e-3*0.5*0.3],[0.2*0.5*0.3, 0.5**2]])
        p_num = MultivariateNumericalDistribution.from_pd(p)
        npt.assert_array_equal(p.central_value, p_num.central_value)
        npt.assert_array_equal(p.support, p_num.support)
        npt.assert_array_almost_equal(p.logpdf([1.6, 2.5]), p_num.logpdf([1.6, 2.5]), decimal=2)
        npt.assert_array_almost_equal(p.logpdf([2.33, 7]), p_num.logpdf([2.33, 7]), decimal=2)

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
        # check for addition of central values
        p_comb = convolve_distributions([p_1, p_x], central_values='sum')
        self.assertIsInstance(p_comb, NormalDistribution)
        self.assertAlmostEqual(p_comb.central_value, 24.7)
        self.assertEqual(p_comb.standard_deviation, sqrt(0.346**2+2.463**2))

    def test_convolve_delta(self):
        p_1 = DeltaDistribution(12.4)
        p_2 = NormalDistribution(12.4, 2.463)
        p_x = DeltaDistribution(12.3)
        from flavio.statistics.probability import convolve_distributions
        with self.assertRaises(NotImplementedError):
            convolve_distributions([p_1, p_x], central_values='sum')
        with self.assertRaises(AssertionError):
            convolve_distributions([p_x, p_2])
        p_comb = convolve_distributions([p_1, p_2])
        self.assertIsInstance(p_comb, NormalDistribution)
        self.assertEqual(p_comb.central_value, 12.4)
        self.assertEqual(p_comb.standard_deviation, 2.463)

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
        # same again for addition
        p_1 = NumericalDistribution.from_pd(NormalDistribution(-986, 0.346))
        p_2 = NumericalDistribution.from_pd(NormalDistribution(16, 2.463))
        p_3 = NumericalDistribution.from_pd(NormalDistribution(107, 1.397))
        conv_p_12 = _convolve_numerical([p_1, p_2], central_values='sum')
        comb_p_12 = NormalDistribution(-970, sqrt(0.346**2 + 2.463**2))
        conv_p_123 = _convolve_numerical([p_1, p_2, p_3], central_values='sum')
        comb_p_123 = NormalDistribution(-863, sqrt(0.346**2 + 2.463**2 + 1.397**2))
        x = np.linspace(-10, 10, 10)
        npt.assert_array_almost_equal(conv_p_12.logpdf(x-970), comb_p_12.logpdf(x-970), decimal=1)
        npt.assert_array_almost_equal(conv_p_123.logpdf(x-863), comb_p_123.logpdf(x-863), decimal=1)

    def test_convolve_multivariate_gaussian(self):
        from flavio.statistics.probability import _convolve_multivariate_gaussians
        cov1 = np.array([[(0.2e-3)**2, 0.2e-3*0.5*0.3],[0.2e-3*0.5*0.3, 0.5**2]])
        cov2 = np.array([[0.2**2, 0.5*0.2*0.4], [0.5*0.2*0.4, 0.4**2]])
        cov12 = cov1 + cov2
        c1 = [2, 5]
        c2 = [-100, -250]
        p_11 = MultivariateNormalDistribution(c1, cov1)
        p_12 = MultivariateNormalDistribution(c1, cov2)
        p_22 = MultivariateNormalDistribution(c2, cov2)
        conv_11_12 = convolve_distributions([p_11, p_12])
        self.assertIsInstance(conv_11_12, MultivariateNormalDistribution)
        npt.assert_array_equal(conv_11_12.central_value, [2, 5])
        npt.assert_array_almost_equal(conv_11_12.covariance, cov12, decimal=15)
        with self.assertRaises(AssertionError):
            convolve_distributions([p_11, p_22])
        conv_11_22 = convolve_distributions([p_11, p_22], central_values='sum')
        self.assertIsInstance(conv_11_22, MultivariateNormalDistribution)
        npt.assert_array_almost_equal(conv_11_22.covariance, cov12, decimal=15)
        npt.assert_array_equal(conv_11_22.central_value, [-100+2, -250+5])

    def test_convolve_multivariate_gaussian_numerical(self):
        from flavio.statistics.probability import convolve_distributions
        cov1 = [[(0.1)**2, 0.1*0.5*0.3],[0.1*0.5*0.3, 0.5**2]]
        cov2 = [[0.2**2, 0.5*0.2*0.4], [0.5*0.2*0.4, 0.4**2]]
        c1 = [2, 5]
        c2 = [-100, -250]
        p_11 = MultivariateNormalDistribution(c1, cov1)
        p_12 = MultivariateNormalDistribution(c1, cov2)
        p_22 = MultivariateNormalDistribution(c2, cov2)
        n_11 = MultivariateNumericalDistribution.from_pd(p_11)
        n_12 = MultivariateNumericalDistribution.from_pd(p_12)
        n_22 = MultivariateNumericalDistribution.from_pd(p_22)
        conv_11_12_gauss = convolve_distributions([p_11, p_12])
        conv_11_12 = convolve_distributions([p_11, n_12])
        self.assertIsInstance(conv_11_12, MultivariateNumericalDistribution)
        npt.assert_array_almost_equal(conv_11_12.central_value, [2, 5], decimal=1)
        self.assertAlmostEqual(conv_11_12.logpdf([2.2, 4]),
                               conv_11_12_gauss.logpdf([2.2, 4]), delta=0.1)
        self.assertAlmostEqual(conv_11_12.logpdf([2.2, 6]),
                               conv_11_12_gauss.logpdf([2.2, 6]), delta=0.1)
        self.assertAlmostEqual(conv_11_12.logpdf([1.4, 4]),
                               conv_11_12_gauss.logpdf([1.4, 4]), delta=0.2)
        self.assertAlmostEqual(conv_11_12.logpdf([1.4, 6]),
                               conv_11_12_gauss.logpdf([1.4, 6]), delta=0.1)
        with self.assertRaises(AssertionError):
            convolve_distributions([p_11, n_22])
        conv_11_22 = convolve_distributions([p_11, n_22], central_values='sum')
        conv_11_22_gauss = convolve_distributions([p_11, p_22], central_values='sum')
        self.assertIsInstance(conv_11_22, MultivariateNumericalDistribution)
        npt.assert_array_almost_equal(conv_11_22.central_value, [-100+2, -250+5], decimal=1)
        self.assertAlmostEqual(conv_11_22.logpdf([2.2-100, 4-250]),
                               conv_11_22_gauss.logpdf([2.2-100, 4-250]), delta=0.1)
        self.assertAlmostEqual(conv_11_22.logpdf([1.6-100, 5.5-250]),
                               conv_11_22_gauss.logpdf([1.6-100, 5.5-250]), delta=0.1)

    def test_1d_errors(self):
        p = NormalDistribution(3, 0.2)
        q = NumericalDistribution.from_pd(p)
        self.assertEqual(p.error_left, 0.2)
        self.assertEqual(p.error_right, 0.2)
        self.assertAlmostEqual(q.error_left, 0.2, places=2)
        self.assertAlmostEqual(q.error_right, 0.2, places=2)
        self.assertAlmostEqual(q.get_error_left(method='hpd'), 0.2, places=2)
        self.assertAlmostEqual(q.get_error_left(method='hpd', nsigma=2), 0.4, places=2)
        self.assertAlmostEqual(q.get_error_right(method='hpd'), 0.2, places=2)

        p = AsymmetricNormalDistribution(3, 0.2, 0.5)
        q = NumericalDistribution.from_pd(p)
        self.assertEqual(p.error_left, 0.5)
        self.assertEqual(p.error_right, 0.2)
        self.assertAlmostEqual(q.error_left, 0.5, places=2)
        self.assertAlmostEqual(q.error_right, 0.2, places=2)
        self.assertAlmostEqual(q.get_error_left(method='hpd'), 0.5, places=2)
        self.assertAlmostEqual(q.get_error_right(method='hpd'), 0.2, places=2)
        self.assertAlmostEqual(q.get_error_right(method='hpd', nsigma=2), 0.4, places=2)

        p = DeltaDistribution(3)
        self.assertEqual(p.error_left, 0)
        self.assertEqual(p.error_right, 0)

        p = UniformDistribution(3, 0.4)
        q = NumericalDistribution.from_pd(p)
        self.assertAlmostEqual(p.error_left, 0.4*0.68, places=2)
        self.assertAlmostEqual(p.error_right, 0.4*0.68, places=2)
        self.assertAlmostEqual(q.error_left, 0.4*0.68, places=2)
        self.assertAlmostEqual(q.error_right, 0.4*0.68, places=2)
        self.assertAlmostEqual(q.get_error_left(method='hpd'), 0.4*0.68, places=2)
        self.assertAlmostEqual(q.get_error_right(method='hpd'), 0.4*0.68, places=2)
        self.assertAlmostEqual(q.get_error_right(method='hpd', nsigma=2), 0.4*0.95, places=2)

        p = HalfNormalDistribution(3, +0.5)
        q = NumericalDistribution.from_pd(p)
        self.assertEqual(p.error_left, 0)
        self.assertEqual(p.error_right, 0.5)
        self.assertAlmostEqual(q.error_left, 0, places=2)
        self.assertAlmostEqual(q.error_right, 0.5, places=2)
        # this does not work (returns nan)
        self.assertTrue(np.isnan(q.get_error_left(method='hpd')))
        self.assertTrue(np.isnan(q.get_error_right(method='hpd')))
        # this works
        self.assertAlmostEqual(q.get_error_right(method='limit'), 0.5, places=2)

        p = HalfNormalDistribution(3, -0.5)
        q = NumericalDistribution.from_pd(p)
        self.assertEqual(p.error_left, 0.5)
        self.assertEqual(p.error_right, 0)
        self.assertAlmostEqual(q.error_left, 0.5, places=2)
        self.assertAlmostEqual(q.error_right, 0, places=2)
        # this does not work (returns nan)
        self.assertTrue(np.isnan(q.get_error_left(method='hpd')))
        self.assertTrue(np.isnan(q.get_error_right(method='hpd')))
        # this works
        self.assertAlmostEqual(q.get_error_left(method='limit'), 0.5, places=2)
        self.assertAlmostEqual(q.get_error_left(method='limit', nsigma=2), 1, places=2)

    def test_multivariate_exclude(self):
        c2 = np.array([1e-3, 2])
        c3 = np.array([1e-3, 2, 0.4])
        cov22 = np.array([[(0.2e-3)**2, 0.2e-3*0.5*0.3],[0.2e-3*0.5*0.3, 0.5**2]])
        cov33 = np.array([[(0.2e-3)**2, 0.2e-3*0.5*0.3 , 0],[0.2e-3*0.5*0.3, 0.5**2, 0.01], [0, 0.01, 0.1**2]])
        pdf1 = NormalDistribution(2, 0.5)
        pdf2 = MultivariateNormalDistribution(c2, cov22)
        pdf3 = MultivariateNormalDistribution(c3, cov33)
        self.assertEqual(pdf2.logpdf([1.1e-3, 2.4]), pdf3.logpdf([1.1e-3, 2.4], exclude=2))
        self.assertEqual(pdf1.logpdf(2.4), pdf3.logpdf([2.4], exclude=(0,2)))
        with self.assertRaises(ValueError):
            # dimensions don't match
            self.assertEqual(pdf2.logpdf([1.1e-3, 2.4]), pdf3.logpdf([1.1e-3, 2.4, 0.2], exclude=2))

    def test_gaussian_kde(self):
        # check that a random Gaussian is reproduced correctly
        np.random.seed(42)
        dat = np.random.normal(117, 23, size=100)
        kde = GaussianKDE(dat)
        norm = scipy.stats.norm(117, 23)
        x = np.linspace(117-23, 117+23, 10)
        npt.assert_array_almost_equal(kde.pdf(x)/norm.pdf(x), np.ones(10), decimal=1)
        # check scott's factor
        self.assertAlmostEqual(kde.bandwidth, 0.4*23, delta=0.4*23*0.1*2)

    def test_vectorize(self):
        # check that all logpdf methods work on arrays as well
        np.random.seed(42)
        xr = np.random.rand(10)
        d = UniformDistribution(0, 1)
        self.assertEqual(d.logpdf(xr).shape, (10,))
        d = DeltaDistribution(1)
        lpd = d.logpdf([2,3,4,5,1,1,3,6,1,3,5,1])
        npt.assert_array_equal(lpd, [-np.inf, -np.inf, -np.inf, -np.inf,
                                     0, 0, -np.inf, -np.inf, 0,
                                     -np.inf, -np.inf, 0 ])
        d = NormalDistribution(0, 1)
        self.assertEqual(d.logpdf(xr).shape, (10,))
        d = AsymmetricNormalDistribution(0, 1, 0.5)
        self.assertEqual(d.logpdf(xr).shape, (10,))
        d = HalfNormalDistribution(0, 1)
        self.assertEqual(d.logpdf(xr).shape, (10,))
        d = GammaDistributionPositive(1, 0, 3)
        self.assertEqual(d.logpdf(xr).shape, (10,))
        d = NumericalDistribution.from_pd(NormalDistribution(0, 1))
        self.assertEqual(d.logpdf(xr).shape, (10,))
        d = MultivariateNormalDistribution([1, 2, 3], np.eye(3))
        xr3 = np.random.rand(10, 3)
        xr2 = np.random.rand(10, 2)
        self.assertEqual(d.logpdf(xr3[0]).shape, ())
        self.assertEqual(d.logpdf(xr3).shape, (10,))
        self.assertEqual(d.logpdf(xr2[0], exclude=(0,)).shape, ())
        self.assertEqual(d.logpdf(xr2, exclude=(0,)).shape, (10,))
        self.assertEqual(d.logpdf(xr[0], exclude=(0, 1)).shape, ())
        self.assertEqual(d.logpdf(xr, exclude=(0, 1)).shape, (10,))
        xi = [np.linspace(-1,1,5), np.linspace(-1,1,6), np.linspace(-1,1,7)]
        y = np.random.rand(5,6,7)
        d = MultivariateNumericalDistribution(xi, y)
        xr3 = np.random.rand(10, 3)
        xr2 = np.random.rand(10, 2)
        self.assertEqual(d.logpdf(xr3[0]).shape, ())
        self.assertEqual(d.logpdf(xr3).shape, (10,))
        self.assertEqual(d.logpdf(xr2[0], exclude=(0,)).shape, ())
        self.assertEqual(d.logpdf(xr2, exclude=(0,)).shape, (10,))
        self.assertEqual(d.logpdf(xr[0], exclude=(0, 1)).shape, ())
        self.assertEqual(d.logpdf(xr, exclude=(0, 1)).shape, (10,))

    def test_repr(self):
        """Test the __repr__ method of all PDs"""

        fsp = 'flavio.statistics.probability.'
        self.assertEqual(repr(NormalDistribution(1, 2)),
                         fsp + 'NormalDistribution(1, 2)')
        self.assertEqual(repr(HalfNormalDistribution(1, -2)),
                         fsp + 'HalfNormalDistribution(1, -2)')
        self.assertEqual(repr(AsymmetricNormalDistribution(1, 2, 3.)),
                         fsp + 'AsymmetricNormalDistribution(1, 2, 3.0)')
        self.assertEqual(repr(DeltaDistribution(-3.)),
                         fsp + 'DeltaDistribution(-3.0)')
        self.assertEqual(repr(UniformDistribution(1, 2)),
                         fsp + 'UniformDistribution(1, 2)')
        self.assertEqual(repr(GaussianUpperLimit(1e-9, 0.95)),
                         fsp + 'GaussianUpperLimit(1e-09, 0.95)')
        self.assertEqual(repr(GammaDistribution(5, -2, 1.5)),
                         fsp + 'GammaDistribution(5, -2, 1.5)')
        self.assertEqual(repr(GammaDistributionPositive(5, -2, 1.5)),
                         fsp + 'GammaDistributionPositive(5, -2, 1.5)')
        self.assertEqual(repr(GammaUpperLimit(15, 10, 1e-9, 0.95)),
                         fsp + 'GammaUpperLimit(15, 10, 1e-09, 0.95)')
        self.assertEqual(repr(GeneralGammaUpperLimit(1e-9, 0.95, counts_total=15, counts_background=10, background_variance=0.2)),
                         fsp + 'GeneralGammaUpperLimit(1e-09, 0.95, counts_total=15, counts_signal=5, background_variance=0.2)')
        self.assertEqual(repr(MultivariateNormalDistribution([1., 2], [[2, 0.1], [0.1, 2]])),
                         fsp + 'MultivariateNormalDistribution([1.0, 2], [[2, 0.1], [0.1, 2]])')
        self.assertEqual(repr(NumericalDistribution([1., 2], [3, 4.])),
                         fsp + 'NumericalDistribution([1.0, 2], [3, 4.0])')
        self.assertEqual(repr(GaussianKDE([1, 2, 3], 0.1)),
                         fsp + 'GaussianKDE([1, 2, 3], 0.1, 3)')
        self.assertEqual(repr(KernelDensityEstimate([1, 2, 3], NormalDistribution(0, 0.5))),
                         fsp + 'KernelDensityEstimate([1, 2, 3], ' + fsp + 'NormalDistribution(0, 0.5), 3)')
        self.assertEqual(repr(MultivariateNumericalDistribution([[1., 2], [10., 20]], [[3, 4.],[5, 6.]], [2, 3])),
                         fsp + 'MultivariateNumericalDistribution([[1.0, 2.0], [10.0, 20.0]], [[3.0, 4.0], [5.0, 6.0]], [2, 3])')


    def test_class_string(self):
        class_from_string_old = {
         'delta': DeltaDistribution,
         'uniform': UniformDistribution,
         'normal': NormalDistribution,
         'asymmetric_normal': AsymmetricNormalDistribution,
         'half_normal': HalfNormalDistribution,
         'gaussian_upper_limit': GaussianUpperLimit,
         'gamma': GammaDistribution,
         'gamma_positive': GammaDistributionPositive,
         'gamma_upper_limit': GammaUpperLimit,
         'general_gamma_upper_limit': GeneralGammaUpperLimit,
         'numerical': NumericalDistribution,
         'multivariate_normal': MultivariateNormalDistribution,
         'multivariate_numerical': MultivariateNumericalDistribution,
         'gaussian_kde': GaussianKDE,
        }
        for k, v in class_from_string_old.items():
            self.assertEqual(v.class_to_string(), k)
            self.assertEqual(string_to_class(k), v)
            self.assertEqual(string_to_class(v.__name__), v)
        self.assertEqual(class_from_string_old,
                        {k: v for k, v in class_from_string.items()
                         if v != KernelDensityEstimate
                         and v != LogNormalDistribution},
                         msg="Failed for {}".format(k))

    def test_get_yaml(self):
        """Test the test_get_yaml method of all PDs"""

        self.assertEqual(yaml.load(NormalDistribution(1, 2).get_yaml()),
                         {'distribution': 'normal',
                         'central_value': 1,
                         'standard_deviation': 2})
        self.assertEqual(yaml.load(HalfNormalDistribution(1, -2).get_yaml()),
                         {'distribution': 'half_normal',
                         'central_value': 1,
                         'standard_deviation': -2})
        self.assertEqual(yaml.load(AsymmetricNormalDistribution(1, 2, 3.).get_yaml()),
                         {'distribution': 'asymmetric_normal',
                         'central_value': 1,
                         'right_deviation': 2,
                         'left_deviation': 3.})
        self.assertEqual(yaml.load(MultivariateNormalDistribution([1., 2], [[4, 0.2], [0.2, 4]]).get_yaml()),
                         {'distribution': 'multivariate_normal',
                         'central_value': [1., 2],
                         'covariance': [[4, 0.2], [0.2, 4]],
                         'standard_deviation': [2, 2],
                         'correlation': [[1, 0.05], [0.05, 1]],
                         })
        self.assertEqual(yaml.load(KernelDensityEstimate([1, 2, 3], NormalDistribution(0, 0.5)).get_yaml()),
                         {'distribution': 'kernel_density_estimate',
                         'data': [1, 2, 3],
                         'kernel':  {'distribution': 'normal',
                          'central_value': 0,
                          'standard_deviation': 0.5},
                         'n_bins': 3})
        self.assertEqual(yaml.load(MultivariateNumericalDistribution([[1., 2], [10., 20]], [[3, 4.],[5, 6.]], [2, 3]).get_yaml()),
                         {'distribution': 'multivariate_numerical',
                         'xi': [[1.0, 2.0], [10.0, 20.0]],
                         'y': [[3.0, 4.0], [5.0, 6.0]],
                         'central_value': [2, 3]})

    def test_get_dict(self):
        ps = [
            NormalDistribution(1, 2),
            HalfNormalDistribution(1, -2),
            AsymmetricNormalDistribution(1, 2, 3.),
            DeltaDistribution(-3.),
            UniformDistribution(1, 2),
            GaussianUpperLimit(1e-9, 0.95),
            GammaDistribution(5, -2, 1.5),
            GammaDistributionPositive(5, -2, 1.5),
            GammaUpperLimit(15, 10, 1e-9, 0.95),
            GeneralGammaUpperLimit(1e-9, 0.95, counts_total=15, counts_background=10, background_variance=0.2),
            MultivariateNormalDistribution([1., 2], [[2, 0.1], [0.1, 2]]),
            NumericalDistribution([1., 2], [3, 4.]),
            GaussianKDE([1, 2, 3], 0.1),
            KernelDensityEstimate([1, 2, 3], NormalDistribution(0, 0.5)),
            MultivariateNumericalDistribution([[1., 2], [10., 20]], [[3, 4.],[5, 6.]], [2, 3])
        ]
        for p in ps:
            # try instantiating a class by feeding the get_dict to __init__
            d = p.get_dict()
            pnew = p.__class__(**d)
            # check if the new class is the same as the old
            self.assertEqual(repr(pnew), repr(p))
            self.assertEqual(pnew.get_yaml(), p.get_yaml())

    def test_dict2dist(self):
        d = [
            {'distribution': 'normal', 'central_value': 1, 'standard_deviation': 0.2},
            {'distribution': 'uniform', 'central_value': 2, 'half_range': 1}
        ]
        p = dict2dist(d)
        self.assertEqual(repr(p[0]), repr(NormalDistribution(1.0, 0.2)))
        self.assertEqual(repr(p[1]), repr(UniformDistribution(2.0, 1.0)))
        p = dict2dist(d[0])
        self.assertEqual(repr(p[0]), repr(NormalDistribution(1.0, 0.2)))

    def test_mvnormal_correlation(self):
        p1 = MultivariateNormalDistribution([0, 0], [[1, 1.5], [1.5, 4]])
        p2 = MultivariateNormalDistribution([0, 0],
                                        standard_deviation=[1, 2],
                                        correlation=[[1, 0.75], [0.75, 1]])
        for p in [p1, p2]:
            npt.assert_array_equal(p.covariance, np.array([[1, 1.5], [1.5, 4]]))
            npt.assert_array_equal(p.standard_deviation, np.array([1, 2]))
            npt.assert_array_equal(p.correlation, np.array([[1, 0.75], [0.75, 1]]))
        with self.assertRaises(ValueError):
            MultivariateNormalDistribution([0, 0], correlation=[[1, 0.75], [0.75, 1]])


class TestCombineDistributions(unittest.TestCase):

    def test_combine_normal(self):
        p_1 = NormalDistribution(5, 0.2)
        p_2 = NormalDistribution(4, 0.3)
        p_comb = combine_distributions([p_1, p_2])
        self.assertIsInstance(p_comb, NormalDistribution)
        s = np.array([0.2, 0.3])
        c = np.array([5, 4])
        w = 1 / s**2  # weights
        s_comb = sqrt(1 / np.sum(w))
        c_comb = np.sum(c * w) / np.sum(w)
        self.assertEqual(p_comb.central_value, c_comb)
        self.assertEqual(p_comb.standard_deviation, s_comb)

    def test_combine_delta(self):
        pd_1 = DeltaDistribution(12.5)
        pd_2 = DeltaDistribution(12.3)
        pn = NormalDistribution(12.4, 2.463)
        with self.assertRaises(ValueError):
            combine_distributions([pd_1, pd_2])
        for pd in [pd_1, pd_2]:
            p_comb = combine_distributions([pd, pn])
            self.assertIsInstance(p_comb, DeltaDistribution)
            self.assertEqual(p_comb.central_value, pd.central_value)

    def test_combine_numerical(self):
        p_1 = NumericalDistribution.from_pd(NormalDistribution(5, 0.2))
        p_2 = NumericalDistribution.from_pd(NormalDistribution(4, 0.3))
        p_comb = combine_distributions([p_1, p_2])
        self.assertIsInstance(p_comb, NumericalDistribution)
        s = np.array([0.2, 0.3])
        c = np.array([5, 4])
        w = 1 / s**2  # weights
        s_comb = sqrt(1 / np.sum(w))
        c_comb = np.sum(c * w) / np.sum(w)
        self.assertAlmostEqual(p_comb.central_value, c_comb, places=2)
        self.assertAlmostEqual(p_comb.error_left, s_comb, places=2)
        self.assertAlmostEqual(p_comb.error_right, s_comb, places=2)
