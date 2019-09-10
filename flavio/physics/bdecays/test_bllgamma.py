import unittest
import numpy as np
from .bllgamma import *
from flavio.physics.eft import WilsonCoefficients
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict
from flavio.parameters import default_parameters
import flavio
import copy

c = copy.deepcopy(default_parameters)
c.set_constraint('f_Bs', '0.2303')
c.set_constraint('f_B0', '0.192')
par = c.get_central_all()

wc_obj = WilsonCoefficients()

class TestBllgamma(unittest.TestCase):
    def test_bllgamma(self):
        # numerical comparison to the code used for 1708.02649
        self.assertAlmostEqual(bllg_dbrdq2_int(0.04465, 8.641, wc_obj, par, 'Bs', 'mu')*10**9, 8.4, places=0)
        self.assertAlmostEqual(bllg_dbrdq2_int(15.84, 28.27, wc_obj, par, 'Bs', 'mu')*10**9, 1.7, places=0)
        self.assertAlmostEqual(bllg_dbrdq2_int(0.04465, 8.641, wc_obj, par, 'Bs', 'e')*10**9, 9.0, places=0)
        self.assertAlmostEqual(bllg_dbrdq2_int(15.84, 28.27, wc_obj, par, 'Bs', 'e')*10**9, 1.4, places=0)
        

