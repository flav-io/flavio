import unittest
import numpy as np
from .bpnunu import *
import flavio

constraints = flavio.default_parameters
wc_obj = flavio.WilsonCoefficients()
par = constraints.get_central_all()

class TestBPnunu(unittest.TestCase):
    def test_bknunu(self):
        # just check that the SM prediction of BR(B->Knunu) is OK
        self.assertAlmostEqual(
            flavio.sm_prediction('BR(B+->Knunu)')/4.68e-6,
            1, delta=0.2)
        # just check the other stuff doesn't raise errors
        flavio.sm_prediction('dBR/dq2(B0->Knunu)', 11)
        flavio.sm_prediction('<dBR/dq2>(B0->Knunu)', 11, 13)
