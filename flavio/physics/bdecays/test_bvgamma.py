import unittest
import numpy as np
from .bvgamma import *
from flavio.physics.bdecays.formfactors.b_v import bsz_parameters
from flavio.physics.eft import WilsonCoefficients
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict
from flavio.parameters import default_parameters
import copy

s = 1.519267515435317e+24

c = copy.copy(default_parameters)
bsz_parameters.bsz_load_v1_lcsr(c)
par = c.get_central_all()

wc_obj = WilsonCoefficients()
wc = wctot_dict(wc_obj, 'bsee', 4.2, par)

class TestBVgamma(unittest.TestCase):
    def test_bksgamma(self):
        # just some trivial tests to see if calling the functions raises an error
        prefactor(par, 'B0', 'K*0')
        # S and ACP should vanish as there are no strong phases or power corrections yet
        self.assertEqual(ACP(wc_obj, par, 'B0', 'K*0'),   0.)
        self.assertEqual(S(wc_obj, par, 'B0', 'K*0'),   0.)
        # # rough numerical comparison of CP-averaged observables to 1503.05534v1
        # # FIXME this should work much better with NLO corrections ...
        self.assertAlmostEqual(BR(wc_obj, par, 'B0', 'K*0')*1e4/4.22, 1, places=-1)
        self.assertAlmostEqual(BR(wc_obj, par, 'B+', 'K*+')*1e4/4.42, 1, places=-1)
