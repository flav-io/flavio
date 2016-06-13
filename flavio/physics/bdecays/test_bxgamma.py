import unittest
import numpy as np
import flavio

wc_obj = flavio.WilsonCoefficients()
par = flavio.default_parameters
par_dict = par.get_central_all()

class TestBXgamma(unittest.TestCase):
    def test_bxgamma(self):
        # compare SM predictions to arXiv:1503.01789
        self.assertAlmostEqual(1e4*flavio.sm_prediction('BR(B->Xsgamma)'),
                               3.36,
                               delta=0.2)
        self.assertAlmostEqual(1e5*flavio.sm_prediction('BR(B->Xdgamma)'),
                               1.73,
                               delta=0.2)
