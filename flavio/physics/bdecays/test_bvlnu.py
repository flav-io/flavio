import unittest
import numpy as np
from .bvlnu import *
from flavio.physics.bdecays.formfactors.b_v import bsz_parameters
from flavio.physics.eft import WilsonCoefficients
from flavio.parameters import default_parameters
import copy

s = 1.519267515435317e+24

c = copy.copy(default_parameters)
bsz_parameters.bsz_load_v1_lcsr(c)
par = c.get_central_all()
wc_obj = WilsonCoefficients()

class TestBVll(unittest.TestCase):
    def test_brhoee(self):
        # just some trivial tests to see if calling the functions raises an error
        q2 = 3.5
        dBRdq2(q2, wc_obj, par, 'B0', 'rho+', 'e')
