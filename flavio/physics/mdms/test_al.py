import unittest
import flavio
from wilson import Wilson
from math import sqrt, pi
import numpy as np


par = flavio.default_parameters.get_central_all()
amu_SM = 116591810e-11
atau_SM = 117721e-8
ae_SM = 0.00115965218157


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

    def test_ae_SM(self):
        self.assertAlmostEqual(flavio.sm_prediction('a_e') / ae_SM, 1)
        pd = flavio.combine_measurements('a_e')
        ae_exp = pd.central_value
        ae_err_exp = pd.error_left
        np.random.seed(17)
        ae_err_sm = flavio.sm_uncertainty('a_e')
        # check that there is a -2.3 sigma tension, see 1804.07409 p. 13
        self.assertAlmostEqual((ae_exp - ae_SM) / sqrt(ae_err_sm**2 + ae_err_exp**2), -2.3, delta=0.5)
