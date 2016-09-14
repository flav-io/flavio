import unittest
import numpy as np
import flavio


class TestBXll(unittest.TestCase):
    def test_bxll(self):
        # compare SM predictions to arXiv:1503.04849
        self.assertAlmostEqual(1e6*flavio.sm_prediction('<BR>(B->Xsmumu)', 1, 6)/1.62,
                               1, delta=0.1)
        self.assertAlmostEqual(1e7*flavio.sm_prediction('<BR>(B->Xsmumu)', 14.4, 25)/2.53,
                               1, delta=0.2)
