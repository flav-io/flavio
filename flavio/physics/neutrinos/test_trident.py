import unittest
import flavio
from wilson import Wilson
from math import sqrt


# constants used in the tests
GF = flavio.default_parameters.get_central('GF')
s2w = flavio.default_parameters.get_central('s2w')
pre = -4 * GF / sqrt(2)
v = 1 / sqrt(sqrt(2) * GF)


class TestNeutrinoTrident(unittest.TestCase):
    def test_trident_sm(self):
        self.assertEqual(flavio.sm_prediction('R_trident'), 1)

    def test_trident_np_1(self):
        # mimic the SM to increase the rate 4x
        w = Wilson({'ll_2222': pre * (1 - 1 / 2 + s2w) / 2, 'le_2222': pre * s2w}, 91.1876, 'SMEFT', 'Warsaw')
        self.assertAlmostEqual(flavio.np_prediction('R_trident', w), 4, delta=0.03)

    def test_trident_np_2(self):
        # mimic the SM but with taus or electrons to increase the rate 2x
        w = Wilson({'ll_1222': pre * (1 - 1 / 2 + s2w), 'le_1222': pre * s2w}, 91.1876, 'SMEFT', 'Warsaw')
        self.assertAlmostEqual(flavio.np_prediction('R_trident', w), 2, delta=0.015)
        w = Wilson({'ll_2223': pre * (1 - 1 / 2 + s2w), 'le_2322': pre * s2w}, 91.1876, 'SMEFT', 'Warsaw')
        self.assertAlmostEqual(flavio.np_prediction('R_trident', w), 2, delta=0.015)

    def test_trident_zprime(self):
        # reproduce eq. (11) in arXiv:1406.2332
        vp = 1000
        CV = -pre * v**2 / vp**2
        w = Wilson({'ll_2222': -CV / 2 / 2, 'le_2222':  -CV / 2}, 91.1876, 'SMEFT', 'Warsaw')
        R = (1 + (1 + 4 * s2w + 2 * v**2 / vp**2)**2) / (1 + (1 + 4 * s2w)**2)
        self.assertAlmostEqual(flavio.np_prediction('R_trident', w), R, delta=0.005)
