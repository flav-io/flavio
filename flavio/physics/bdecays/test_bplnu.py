import unittest
import numpy as np
import flavio

class TestBVll(unittest.TestCase):
    def test_bpienu(self):
        pass #TODO

    def test_lfratios(self):
        self.assertAlmostEqual(
            flavio.sm_prediction('<Rmue>(B->Dlnu)', 0.5, 5), 1, delta=0.01)
        self.assertAlmostEqual(
            flavio.sm_prediction('<Rmue>(B->pilnu)', 0.5, 5), 1, delta=0.01)
        self.assertAlmostEqual(
            flavio.sm_prediction('Rmue(B->Dlnu)'), 1, delta=0.01)
        self.assertAlmostEqual(
            flavio.sm_prediction('Rmue(B->pilnu)'), 1, delta=0.01)
        # for the taus, just make sure no error is raised
        flavio.sm_prediction('Rtaumu(B->pilnu)')
        flavio.sm_prediction('<Rtaumu>(B->Dlnu)', 15, 16)
