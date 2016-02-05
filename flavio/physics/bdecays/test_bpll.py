import unittest
import numpy as np
from .bpll import *
from flavio.physics.eft import WilsonCoefficients
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict
import copy
from flavio.parameters import default_parameters
from flavio.classes import Parameter

c = copy.copy(default_parameters)
try:
    Parameter('B->K BCL a0_f+')
    Parameter('B->K BCL a0_f0')
    Parameter('B->K BCL a0_fT')
    Parameter('B->K BCL a1_f+')
    Parameter('B->K BCL a1_f0')
    Parameter('B->K BCL a1_fT')
    Parameter('B->K BCL a2_f+')
    Parameter('B->K BCL a2_f0')
    Parameter('B->K BCL a2_fT')
except:
    pass
c.set_constraint('B->K BCL a0_f+', 0.466)
c.set_constraint('B->K BCL a1_f+', -0.885)
c.set_constraint('B->K BCL a2_f+', -0.213)
c.set_constraint('B->K BCL a0_f0', 0.292)
c.set_constraint('B->K BCL a1_f0', 0.281)
c.set_constraint('B->K BCL a2_f0', 0.150)
c.set_constraint('B->K BCL a0_fT', 0.460)
c.set_constraint('B->K BCL a1_fT', -1.089)
c.set_constraint('B->K BCL a2_fT', -1.114)

par = c.get_central_all()

wc_obj = WilsonCoefficients()
wc = wctot_dict(wc_obj, 'bsmumu', 4.2, par)

class TestBPll(unittest.TestCase):
    def test_bkll(self):
        # rough numerical test for branching ratio at high q^2 comparing to 1510.02349
        self.assertAlmostEqual(bpll_dbrdq2(16., wc_obj, par, 'B+', 'K+', 'mu')*1e8/(4.615/2.), 1, places=1)
