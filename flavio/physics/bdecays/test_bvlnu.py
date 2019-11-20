import unittest
import numpy as np
from .bvlnu import *
from flavio.physics.bdecays.formfactors.b_v import bsz_parameters
from flavio.physics.eft import WilsonCoefficients
from flavio.parameters import default_parameters
import flavio
from math import pi

constraints = default_parameters
wc_obj = WilsonCoefficients()
par = constraints.get_central_all()

class TestBVll(unittest.TestCase):
    def test_bdsenu(self):
        Vcb = flavio.default_parameters.get_central('Vcb')
        # assert that total BR is in the ballpark of the experimental number
        self.assertAlmostEqual(
            flavio.sm_prediction('BR(B+->D*lnu)') / 5.41e-2 * 0.04**2 / Vcb**2, 1, delta=0.1)


    def test_brhoee(self):
        q2 = 3.5
        self.assertEqual(
            dBRdq2(q2, wc_obj, par, 'B0', 'rho+', 'e', A=None),
            flavio.Observable["dBR/dq2(B0->rhoenu)"].prediction_central(constraints, wc_obj, q2=q2) )

    def test_decays(self):
        # just check if any of the modes raises an exception
        flavio.Observable["dBR/dq2(B0->rhoenu)"].prediction_central(constraints, wc_obj, q2=3)
        flavio.Observable["dBR/dq2(B+->rhoenu)"].prediction_central(constraints, wc_obj, q2=3)
        flavio.Observable["dBR/dq2(B0->D*enu)"].prediction_central(constraints, wc_obj, q2=3)
        flavio.Observable["dBR/dq2(B+->D*enu)"].prediction_central(constraints, wc_obj, q2=3)
        flavio.Observable["dBR/dq2(B+->omegaenu)"].prediction_central(constraints, wc_obj, q2=3)
        flavio.Observable["dBR/dq2(Bs->K*enu)"].prediction_central(constraints, wc_obj, q2=3)
        flavio.Observable["dBR/dcl(B0->rhoenu)"].prediction_central(constraints, wc_obj, cl=0.5)
        flavio.Observable["dBR/dcl(B+->rhoenu)"].prediction_central(constraints, wc_obj, cl=0.5)
        flavio.Observable["dBR/dcl(B0->D*enu)"].prediction_central(constraints, wc_obj, cl=0.5)
        flavio.Observable["dBR/dcl(B+->D*enu)"].prediction_central(constraints, wc_obj, cl=0.5)
        flavio.Observable["dBR/dcl(B+->omegaenu)"].prediction_central(constraints, wc_obj, cl=0.5)
        flavio.Observable["dBR/dcl(Bs->K*enu)"].prediction_central(constraints, wc_obj, cl=0.5)
        flavio.Observable["dBR/dcV(B0->rhoenu)"].prediction_central(constraints, wc_obj, cV=0.5)
        flavio.Observable["dBR/dcV(B+->rhoenu)"].prediction_central(constraints, wc_obj, cV=0.5)
        flavio.Observable["dBR/dcV(B0->D*enu)"].prediction_central(constraints, wc_obj, cV=0.5)
        flavio.Observable["dBR/dcV(B+->D*enu)"].prediction_central(constraints, wc_obj, cV=0.5)
        flavio.Observable["dBR/dcV(B+->omegaenu)"].prediction_central(constraints, wc_obj, cV=0.5)
        flavio.Observable["dBR/dcV(Bs->K*enu)"].prediction_central(constraints, wc_obj, cV=0.5)
        flavio.Observable["dBR/dphi(B0->rhoenu)"].prediction_central(constraints, wc_obj, phi=1.5)
        flavio.Observable["dBR/dphi(B+->rhoenu)"].prediction_central(constraints, wc_obj, phi=1.5)
        flavio.Observable["dBR/dphi(B0->D*enu)"].prediction_central(constraints, wc_obj, phi=1.5)
        flavio.Observable["dBR/dphi(B+->D*enu)"].prediction_central(constraints, wc_obj, phi=1.5)
        flavio.Observable["dBR/dphi(B+->omegaenu)"].prediction_central(constraints, wc_obj, phi=1.5)
        flavio.Observable["dBR/dphi(Bs->K*enu)"].prediction_central(constraints, wc_obj, phi=1.5)

    def test_binned(self):
        # this is all the total BR, calulated in 4 different ways
        self.assertAlmostEqual(
        flavio.Observable["BR(B0->D*enu)"].prediction_central(
                                constraints, wc_obj),
        flavio.Observable["<BR>/<cl>(B0->D*enu)"].prediction_central(
                                constraints, wc_obj, clmin=-1, clmax=1))
        self.assertAlmostEqual(
        flavio.Observable["BR(B0->D*enu)"].prediction_central(
                                constraints, wc_obj),
        flavio.Observable["<BR>/<cV>(B0->D*enu)"].prediction_central(
                                constraints, wc_obj, cVmin=-1, cVmax=1))
        self.assertAlmostEqual(
        flavio.Observable["BR(B0->D*enu)"].prediction_central(
                                constraints, wc_obj),
        flavio.Observable["<BR>/<phi>(B0->D*enu)"].prediction_central(
                                constraints, wc_obj, phimin=-pi, phimax=pi))

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

    def test_BRLT(self):
        self.assertAlmostEqual(flavio.sm_prediction('dBR/dq2(B0->D*enu)', q2=3),
                               flavio.sm_prediction('dBR_L/dq2(B0->D*enu)', q2=3) +
                               flavio.sm_prediction('dBR_T/dq2(B0->D*enu)', q2=3),
                               places=10)

    def test_bdstau_binned(self):
        # for the full kinematical range, it should integrate to 1
        self.assertAlmostEqual(
            flavio.sm_prediction('<BR>/BR(B->D*taunu)', 3.1, 11.7), 1, delta=0.03)

    def test_FL(self):
        self.assertAlmostEqual(flavio.sm_prediction('<FL>(B0->D*taunu)', 3.15, 10.71),
                               0.46, delta=0.04)
        self.assertAlmostEqual(flavio.sm_prediction('FLtot(B0->D*taunu)'),
                               flavio.sm_prediction('<FL>(B0->D*taunu)', 3.15, 10.71),
                               delta=0.001)
