import unittest
import numpy as np
import flavio

wc_sm = flavio.physics.eft.WilsonCoefficients()
wc_np = flavio.physics.eft.WilsonCoefficients()
wc_np.set_initial({'C10_bsemu':4., 'C10_bsmue':2.}, 160.)

class TestLFV(unittest.TestCase):
    def test_lfv(self):
        obs_1 = flavio.classes.Observable["BR(B0->K*emu)"]
        obs_2 = flavio.classes.Observable["BR(B0->K*mue)"]
        self.assertEqual(obs_1.prediction_central(flavio.default_parameters, wc_sm), 0)
        # BR(B->K*emu) should be 4 times larger as Wilson coeff is 2x the mue one
        self.assertAlmostEqual(
            obs_1.prediction_central(flavio.default_parameters, wc_np)
            /obs_2.prediction_central(flavio.default_parameters, wc_np),
            4.,  places=10)
        # test for errors
        flavio.sm_prediction("BR(B+->K*mue)")
        flavio.sm_prediction("BR(B0->rhotaue)")
        flavio.sm_prediction("BR(B+->rhotaumu)")
        flavio.sm_prediction("BR(Bs->phimutau)")
