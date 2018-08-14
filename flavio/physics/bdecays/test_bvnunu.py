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

    def test_fl(self):
        # compare to 1409.4557 table 2
        self.assertAlmostEqual(
            flavio.sm_prediction('<FL>(B0->K*nunu)', 0, 27),
            0.47,
            delta=2*0.03)
        self.assertAlmostEqual(
            flavio.sm_prediction('<FL>(B0->K*nunu)', 0, 4),
            0.79,
            delta=2*0.03)
        self.assertAlmostEqual(
            flavio.sm_prediction('<FL>(B0->K*nunu)', 16, 19.25),
            0.32,
            delta=2*0.03)
        # ... and the differential ones
        self.assertAlmostEqual(
            flavio.sm_prediction('FL(B0->K*nunu)', 2),
            0.79,
            delta=3*0.03)
        self.assertAlmostEqual(
            flavio.sm_prediction('FL(B0->K*nunu)', 17.5),
            0.32,
            delta=3*0.03)
