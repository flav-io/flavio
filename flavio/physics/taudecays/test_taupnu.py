import unittest
import numpy as np
import flavio

constraints = flavio.default_parameters
wc_obj = flavio.WilsonCoefficients()
par = constraints.get_central_all()

class TestTauPnu(unittest.TestCase):
    def test_taupnu(self):
        # compare to the experimental values
        self.assertAlmostEqual(
            flavio.Observable.get_instance("BR(tau->pinu)").prediction_central(constraints, wc_obj),
            10.82e-2, delta=2*0.05e-2)
        self.assertAlmostEqual(
            flavio.Observable.get_instance("BR(tau->Knu)").prediction_central(constraints, wc_obj),
            6.96e-3, delta=2*0.1e-3)
