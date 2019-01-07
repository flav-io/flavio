import unittest
import numpy as np
import flavio

constraints = flavio.default_parameters
wc_obj = flavio.WilsonCoefficients()
par = constraints.get_central_all()

class TestKlnu(unittest.TestCase):
    def test_klnu(self):
        # compare to the experimental values
        self.assertAlmostEqual(
            flavio.Observable["BR(K+->enu)"].prediction_central(constraints, wc_obj),
            1.582e-5, delta=4*0.007e-5)
        self.assertAlmostEqual(
            flavio.Observable["BR(K+->munu)"].prediction_central(constraints, wc_obj),
            63.56e-2, delta=8*0.11e-2)

    def test_rklnu(self):
        # compare to 0707.4464
        self.assertAlmostEqual(
            flavio.Observable["Remu(K+->lnu)"].prediction_central(constraints, wc_obj),
            2.477e-5, delta=2*0.001e-5)

    def test_pienu(self):
        # compare to the experimental values
        self.assertAlmostEqual(
            flavio.Observable["BR(pi+->enu)"].prediction_central(constraints, wc_obj),
            1.2352e-4, delta=2*0.0001e-2)

    def test_pimunu(self):
        # compare to the experimental values
        self.assertAlmostEqual(
            flavio.Observable["Gamma(pi+->munu)"].prediction_central(constraints, wc_obj)
            * par['tau_pi+'],
            1, delta=0.01)

    def test_klnu_nu(self):
        wc_sm = flavio.WilsonCoefficients()
        wc_np_mu = flavio.WilsonCoefficients()
        wc_np_mu.set_initial({'CVL_sumunumu': 1}, 4.8)
        wc_np_e = flavio.WilsonCoefficients()
        wc_np_e.set_initial({'CVL_sumunue': 1}, 4.8)
        obs = flavio.Observable["BR(K+->munu)"]
        br_sm = obs.prediction_central(constraints, wc_sm)
        br_mu = obs.prediction_central(constraints, wc_np_mu)
        br_e = obs.prediction_central(constraints, wc_np_e)
        # with interference: (1 + 1)^2 = 4
        self.assertAlmostEqual(br_mu/br_sm, 4, delta=0.06)
        # without interference: 1 + 1 = 2
        self.assertAlmostEqual(br_e/br_sm, 2, delta=0.03)
