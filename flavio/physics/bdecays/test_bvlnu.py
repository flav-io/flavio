import unittest
import numpy as np
from .bvlnu import *
from flavio.physics.bdecays.formfactors.b_v import bsz_parameters
from flavio.physics.eft import WilsonCoefficients
from flavio.parameters import default_parameters
import flavio

constraints = default_parameters
wc_obj = WilsonCoefficients()
par = constraints.get_central_all()

class TestBVll(unittest.TestCase):
    def test_brhoee(self):
        q2 = 3.5
        self.assertEqual(
            dBRdq2(q2, wc_obj, par, 'B0', 'rho+', 'e'),
            flavio.Observable["dBR/dq2(B0->rhoenu)"].prediction_central(constraints, wc_obj, q2=q2) )

    def test_decays(self):
        # just check if any of the modes raises an exception
        flavio.Observable["dBR/dq2(B0->rhoenu)"].prediction_central(constraints, wc_obj, q2=3)
        flavio.Observable["dBR/dq2(B+->rhoenu)"].prediction_central(constraints, wc_obj, q2=3)
        flavio.Observable["dBR/dq2(B0->D*enu)"].prediction_central(constraints, wc_obj, q2=3)
        flavio.Observable["dBR/dq2(B+->D*enu)"].prediction_central(constraints, wc_obj, q2=3)
        flavio.Observable["dBR/dq2(B+->omegaenu)"].prediction_central(constraints, wc_obj, q2=3)
        flavio.Observable["dBR/dq2(Bs->K*enu)"].prediction_central(constraints, wc_obj, q2=3)

    def test_bvlnu_nu(self):
        wc_sm = flavio.WilsonCoefficients()
        wc_np_tau = flavio.WilsonCoefficients()
        wc_np_tau.set_initial({'CVL_butaunutau': 1}, 4.8)
        wc_np_e = flavio.WilsonCoefficients()
        wc_np_e.set_initial({'CVL_butaunue': 1}, 4.8)
        obs = flavio.Observable["BR(B+->rhotaunu)"]
        br_sm = obs.prediction_central(constraints, wc_sm)
        br_tau = obs.prediction_central(constraints, wc_np_tau)
        br_e = obs.prediction_central(constraints, wc_np_e)
        # with interference: (1 + 1)^2 = 4
        self.assertAlmostEqual(br_tau/br_sm, 4, delta=0.04)
        # without interference: 1 + 1 = 2
        self.assertAlmostEqual(br_e/br_sm, 2, delta=0.02)
