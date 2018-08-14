import unittest
import numpy as np
import flavio

constraints = flavio.default_parameters
wc_obj = flavio.WilsonCoefficients()
par = constraints.get_central_all()

class TestDlnu(unittest.TestCase):
    def test_dlnu(self):
        # compare to the experimental values
        self.assertAlmostEqual(
            flavio.Observable["BR(Ds->munu)"].prediction_central(constraints, wc_obj),
            5.56e-3, delta=3*0.25e-3)
        self.assertAlmostEqual(
            flavio.Observable["BR(Ds->taunu)"].prediction_central(constraints, wc_obj),
            5.55e-2, delta=3*0.24e-2)
        self.assertAlmostEqual(
            flavio.Observable["BR(D+->munu)"].prediction_central(constraints, wc_obj),
            3.74e-4, delta=3*0.17e-4)

    def test_dlnu_nu(self):
        wc_sm = flavio.WilsonCoefficients()
        wc_np_mu = flavio.WilsonCoefficients()
        wc_np_mu.set_initial({'CVL_scmunumu': 1}, 4.8)
        wc_np_e = flavio.WilsonCoefficients()
        wc_np_e.set_initial({'CVL_scmunue': 1}, 4.8)
        obs = flavio.Observable["BR(Ds->munu)"]
        br_sm = obs.prediction_central(constraints, wc_sm)
        br_mu = obs.prediction_central(constraints, wc_np_mu)
        br_e = obs.prediction_central(constraints, wc_np_e)
        # with interference: (1 + 1)^2 = 4
        self.assertAlmostEqual(br_mu/br_sm, 4, delta=0.04)
        # without interference: 1 + 1 = 2
        self.assertAlmostEqual(br_e/br_sm, 2, delta=0.02)
