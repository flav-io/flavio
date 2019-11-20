import unittest
import numpy as np
import flavio

class TestBVll(unittest.TestCase):
    def test_bpienu(self):
        Vcb = flavio.default_parameters.get_central('Vcb')
        # assert that total BR is in the ballpark of the experimental number
        self.assertAlmostEqual(
            flavio.sm_prediction('BR(B+->Dlnu)') / 2.29e-2 * 0.04**2 / Vcb**2, 1, delta=0.1)

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

    def test_bdtau_binned(self):
        # for the full kinematical range, it should integrate to 1
        self.assertAlmostEqual(
            flavio.sm_prediction('<BR>/BR(B->Dtaunu)', 3.1, 24), 1, delta=0.01)

    def test_bplnu_nu(self):
        wc_sm = flavio.WilsonCoefficients()
        wc_np_tau = flavio.WilsonCoefficients()
        wc_np_tau.set_initial({'CVL_butaunutau': 1}, 4.8)
        wc_np_e = flavio.WilsonCoefficients()
        wc_np_e.set_initial({'CVL_butaunue': 1}, 4.8)
        obs = flavio.Observable["BR(B+->pitaunu)"]
        constraints = flavio.default_parameters
        br_sm = obs.prediction_central(constraints, wc_sm)
        br_tau = obs.prediction_central(constraints, wc_np_tau)
        br_e = obs.prediction_central(constraints, wc_np_e)
        # with interference: (1 + 1)^2 = 4
        self.assertAlmostEqual(br_tau/br_sm, 4, delta=0.04)
        # without interference: 1 + 1 = 2
        self.assertAlmostEqual(br_e/br_sm, 2, delta=0.02)
