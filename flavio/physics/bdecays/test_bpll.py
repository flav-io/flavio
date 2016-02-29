import unittest
import numpy as np
from .bpll import *
from flavio.physics.eft import WilsonCoefficients
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict
import copy
from flavio.parameters import default_parameters
from flavio.classes import Parameter

c = default_parameters.copy()
c.set_constraint('B->K BCL a0_f+', 0.428)
c.set_constraint('B->K BCL a1_f+', -0.674)
c.set_constraint('B->K BCL a2_f+', -1.12)
c.set_constraint('B->K BCL a0_f0', 0.545)
c.set_constraint('B->K BCL a1_f0', -1.91)
c.set_constraint('B->K BCL a2_f0', 1.83)
c.set_constraint('B->K BCL a0_fT', 0.402)
c.set_constraint('B->K BCL a1_fT', -0.535)
c.set_constraint('B->K BCL a2_fT', -0.286)

par = c.get_central_all()

wc_obj = WilsonCoefficients()
wc = wctot_dict(wc_obj, 'bsmumu', 4.2, par)

class TestBPll(unittest.TestCase):
    def test_bkll(self):
        # rough numerical test for branching ratio at high q^2 to old code
        self.assertAlmostEqual(bpll_dbrdq2(15., wc_obj, par, 'B+', 'K+', 'mu')/2.1824401629030333e-8, 1, delta=0.1)
