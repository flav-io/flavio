import unittest
import numpy as np
from .bpll_lfv import *
from flavio.physics.eft import WilsonCoefficients
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict
import copy
from flavio.parameters import default_parameters
from flavio.classes import Parameter
import flavio

c = default_parameters.copy()
c.set_constraint('B->K BCL a0_f+', 0.428)
c.set_constraint('B->K BCL a1_f+', -0.674)
c.set_constraint('B->K BCL a2_f+', -1.12)
c.set_constraint('B->K BCL a0_f0', 0.545)
c.set_constraint('B->K BCL a1_f0', -1.91)
c.set_constraint('B->K BCL a2_f0', 1.83)
c.set_constraint('B->K BCL a0_fT', 0.402)
c.set_constraint('B->K BCL a1_fT', -0.535)
c.set_constraint('B->K BCL a2_fT', -0.286)

par = c.get_central_all()

wc_obj = WilsonCoefficients()
wc = wctot_dict(wc_obj, 'bsmumu', 4.2, par)

wc_sm = flavio.physics.eft.WilsonCoefficients()
wc_lfv = flavio.physics.eft.WilsonCoefficients()
wc_lfv.set_initial({'C10_bsemu':4., 'C10_bsmue':2.}, 160.)


class TestBPll(unittest.TestCase):


    def test_bpll_lfv(self):
        # test for errors
        self.assertEqual(flavio.sm_prediction('BR(B0->Kemu)'), 0)
        self.assertEqual(flavio.sm_prediction('BR(B+->Ktaumu)'), 0)
        self.assertEqual(flavio.sm_prediction('BR(B+->pitaumu)'), 0)
        self.assertEqual(flavio.sm_prediction('BR(B0->pitaumu)'), 0)
        obs_1 = flavio.classes.Observable["BR(B0->Kemu)"]
        obs_2 = flavio.classes.Observable["BR(B0->Kmue)"]
        self.assertEqual(obs_1.prediction_central(flavio.default_parameters, wc_sm), 0)
        # BR(B->Kemu) should be 4 times larger as Wilson coeff is 2x the mue one
        self.assertAlmostEqual(
            obs_1.prediction_central(flavio.default_parameters, wc_lfv)
            /obs_2.prediction_central(flavio.default_parameters, wc_lfv),
            4.,  places=10)
