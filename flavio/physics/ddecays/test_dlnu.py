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
            flavio.Observable.get_instance("BR(Ds->munu)").prediction_central(constraints, wc_obj),
            5.56e-3, delta=3*0.25e-3)
        self.assertAlmostEqual(
            flavio.Observable.get_instance("BR(Ds->taunu)").prediction_central(constraints, wc_obj),
            5.55e-2, delta=3*0.24e-2)
        self.assertAlmostEqual(
            flavio.Observable.get_instance("BR(D+->munu)").prediction_central(constraints, wc_obj),
            3.74e-4, delta=3*0.17e-4)
