import unittest
import numpy as np
from .bplnu import *
from flavio.physics.bdecays.formfactors.b_v import bsz_parameters
from flavio.physics.eft import WilsonCoefficients
from flavio.parameters import default_parameters
import flavio

constraints = default_parameters
wc_obj = WilsonCoefficients()
par = constraints.get_central_all()


wc_obj = WilsonCoefficients()

class TestBVll(unittest.TestCase):
    def test_bpienu(self):
        q2 = 3.5
        # self.assertEqual(
            # dBRdq2(q2, wc_obj, par, 'B0', 'pi+', 'e'),
            # flavio.Observable.get_instance("dBR/dq2(B0->pienu)").prediction_central(constraints, wc_obj, q2=q2) )
