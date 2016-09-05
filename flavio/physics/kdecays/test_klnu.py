import unittest
import numpy as np
import flavio

constraints = flavio.default_parameters
wc_obj = flavio.WilsonCoefficients()
par = constraints.get_central_all()

class TestKlnu(unittest.TestCase):
    def test_klnu(self):
        Vus = flavio.physics.ckm.get_ckm(par)[0,1]
        # compare to the experimental values
        self.assertAlmostEqual(
            flavio.Observable.get_instance("BR(K+->enu)").prediction_central(constraints, wc_obj),
            1.582e-5, delta=4*0.007e-5)
        self.assertAlmostEqual(
            flavio.Observable.get_instance("BR(K+->munu)").prediction_central(constraints, wc_obj),
            63.56e-2, delta=8*0.11e-2)

    def test_rklnu(self):
        # compare to 0707.4464
        self.assertAlmostEqual(
            flavio.Observable.get_instance("Remu(K+->lnu)").prediction_central(constraints, wc_obj),
            2.477e-5, delta=2*0.001e-5)

    def test_pilnu(self):
        Vus = flavio.physics.ckm.get_ckm(par)[0,1]
        # compare to the experimental values
        self.assertAlmostEqual(
            flavio.Observable.get_instance("BR(pi+->enu)").prediction_central(constraints, wc_obj),
            1.2352e-4, delta=2*0.0001e-2)
