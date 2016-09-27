import unittest
import numpy as np
import flavio
import scipy.stats
from math import pi, sqrt, exp, log

class TestProbability(unittest.TestCase):
    def test_multiv_normal(self):
        # test that the rescaling of the MultivariateNormalDistribution
        # does not affect the log PDF!
        c = np.array([1e-3, 2])
        cov = np.array([[(0.2e-3)**2, 0.2e-3*0.5*0.3],[0.2e-3*0.5*0.3, 0.5**2]])
        pdf = flavio.statistics.probability.MultivariateNormalDistribution(c, cov)
        x=np.array([1.5e-3, 0.8])
        num_lpdf = pdf.logpdf(x)
        ana_lpdf = log(1/sqrt(4*pi**2*np.linalg.det(cov))*exp(-np.dot(np.dot(x-c,np.linalg.inv(cov)),x-c)/2))
        self.assertAlmostEqual(num_lpdf, ana_lpdf, delta=1e-6)

    def test_halfnormal(self):
        pdf_p_1 = flavio.statistics.probability.HalfNormalDistribution(1.7, 0.3)
        pdf_n_1 = flavio.statistics.probability.HalfNormalDistribution(1.7, -0.3)
        pdf_p_2 = flavio.statistics.probability.AsymmetricNormalDistribution(1.7, 0.3, 0.0001)
        pdf_n_2 = flavio.statistics.probability.AsymmetricNormalDistribution(1.7, 0.0001, 0.3)
        self.assertAlmostEqual(pdf_p_1.logpdf(1.99), pdf_p_2.logpdf(1.99), delta=0.001)
        self.assertEqual(pdf_p_1.logpdf(1.55), -np.inf)
        self.assertAlmostEqual(pdf_n_1.logpdf(1.55), pdf_n_2.logpdf(1.55), delta=0.001)
        self.assertEqual(pdf_n_1.logpdf(1.99), -np.inf)

    def test_limit(self):
        p1 = flavio.statistics.probability.GaussianUpperLimit(1.78, 0.68268949)
        p2 = flavio.statistics.probability.HalfNormalDistribution(0, 1.78)
        self.assertAlmostEqual(p1.logpdf(0.237), p2.logpdf(0.237), delta=0.0001)
        self.assertEqual(p2.logpdf(-1), -np.inf)

    def test_numerical(self):
        x = np.arange(-4,6,0.1)
        y = scipy.stats.norm.pdf(x, loc=1)
        y_crazy = 14.7 * y # multiply PDF by crazy number
        p_num = flavio.statistics.probability.NumericalDistribution(x, y_crazy)
        p_norm = flavio.statistics.probability.NormalDistribution(1, 1)
        self.assertAlmostEqual(p_num.logpdf(0.237), p_norm.logpdf(0.237), delta=0.02)
        self.assertAlmostEqual(p_num.logpdf(-2.61), p_norm.logpdf(-2.61), delta=0.02)
        # just check if this raises an error
        p_num.get_random(100)

    def test_convolute(self):
        p_1 = flavio.statistics.probability.NormalDistribution(12.4, 0.346)
        p_2 = flavio.statistics.probability.NormalDistribution(12.4, 2.463)
        p_x = flavio.statistics.probability.NormalDistribution(12.3, 2.463)
        from flavio.statistics.probability import convolute_distributions
        # error if not the same central value:
        with self.assertRaises(AssertionError):
            convolute_distributions([p_1, p_x])
        p_comb = convolute_distributions([p_1, p_2])
        self.assertIsInstance(p_comb, flavio.statistics.probability.NormalDistribution)
        self.assertEqual(p_comb.central_value, 12.4)
        self.assertEqual(p_comb.standard_deviation, sqrt(0.346**2+2.463**2))
