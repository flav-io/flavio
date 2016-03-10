import unittest
import numpy as np
from .bvnunu import *
import flavio

constraints = flavio.default_parameters
wc_obj = flavio.WilsonCoefficients()
par = constraints.get_central_all()

class TestBVnunu(unittest.TestCase):
    def test_bksnunu(self):
        # just check that the SM prediction of BR(B->K*nunu) is OK
        self.assertAlmostEqual(
            flavio.sm_prediction('BR(B0->K*nunu)')/9.48e-6,
            1, delta=0.2)
        # just check the other stuff doesn't raise errors
        flavio.sm_prediction('dBR/dq2(B+->K*nunu)', 11)
        flavio.sm_prediction('<dBR/dq2>(B+->K*nunu)', 11, 13)
