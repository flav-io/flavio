import unittest
import flavio


class TestWDecays(unittest.TestCase):

    def test_sm(self):
        self.assertAlmostEqual(flavio.sm_prediction('BR(W->enu)'), 10.83e-2, delta=0.02e-2)
        self.assertAlmostEqual(flavio.sm_prediction('BR(W->munu)'), 10.83e-2, delta=0.02e-2)
        self.assertAlmostEqual(flavio.sm_prediction('BR(W->taunu)'), 10.83e-2, delta=0.02e-2)
        self.assertAlmostEqual(flavio.sm_prediction('GammaW'), 2.091, delta=0.001)

    def test_np(self):
        v = 246.22
        from wilson import Wilson
        w = Wilson({'phil3_11': -0.5 / v**2}, 91.1876, 'SMEFT', 'Warsaw')
        self.assertAlmostEqual(flavio.np_prediction('BR(W->enu)', w), 10.83e-2 / 2, delta=0.15e-2)
