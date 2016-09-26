import unittest
import numpy as np
import flavio
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
