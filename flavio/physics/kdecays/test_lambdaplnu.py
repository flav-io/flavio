import unittest
import flavio


class TestLambdaplnu(unittest.TestCase):
    def test_lambdaplnu(self):
        gf = flavio.sm_prediction('g1/f1(Lambda->penu)')
        self.assertAlmostEqual(gf, 0.7185, delta=0.0001)
