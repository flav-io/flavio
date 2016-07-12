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
