import unittest
import flavio
from wilson import Wilson
from math import sqrt, pi


par = flavio.default_parameters.get_central_all()
amu_SM = 11659182.3e-10
atau_SM = 117721e-8


class TestGminus2Leptons(unittest.TestCase):
    def test_amu_SM(self):
        self.assertAlmostEqual(flavio.sm_prediction('a_mu') / amu_SM, 1)

    def test_amu_NP(self):
        w = Wilson({'C7_mumu': 1e-3}, 1.0, 'WET-3', 'flavio')
        e = sqrt(4 * pi * par['alpha_e'])
        m = par['m_mu']
        p = 4 * par['GF'] / sqrt(2) * e / 16 / pi**2 * m
        pre = p * 4 * m / e
        a = pre * 1e-3
        self.assertAlmostEqual(flavio.np_prediction('a_mu', w) - amu_SM,
                               a,
                               delta=0.01 * abs(a))

    def test_atau_SM(self):
        self.assertAlmostEqual(flavio.sm_prediction('a_tau') / atau_SM, 1)
