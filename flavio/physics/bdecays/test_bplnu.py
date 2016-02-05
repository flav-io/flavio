import unittest
import numpy as np
from .bplnu import *
from flavio.physics.bdecays.formfactors.b_v import bsz_parameters
from flavio.physics.eft import WilsonCoefficients
import copy
from flavio.parameters import default_parameters
from flavio.classes import Parameter

c = copy.copy(default_parameters)
Parameter('B->D BCL a0_f+')
Parameter('B->D BCL a0_f0')
Parameter('B->D BCL a0_fT')
Parameter('B->D BCL a1_f+')
Parameter('B->D BCL a1_f0')
Parameter('B->D BCL a1_fT')
Parameter('B->D BCL a2_f+')
Parameter('B->D BCL a2_f0')
Parameter('B->D BCL a2_fT')
c.set_constraint('B->D BCL a0_f+', 0.836)
c.set_constraint('B->D BCL a1_f+', -2.66)
c.set_constraint('B->D BCL a2_f+', -0.07)
c.set_constraint('B->D BCL a0_f0', 0.647)
c.set_constraint('B->D BCL a1_f0', 0.27)
c.set_constraint('B->D BCL a2_f0', -0.09)
c.set_constraint('B->D BCL a0_fT', 0)
c.set_constraint('B->D BCL a1_fT', 0)
c.set_constraint('B->D BCL a2_fT', 0)
par = c.get_central_all()

wc_obj = WilsonCoefficients()

class TestBVll(unittest.TestCase):
    def test_brhoee(self):
        # just some trivial tests to see if calling the functions raises an error
        q2 = 3.5
        dBRdq2(q2, wc_obj, par, 'B+', 'D0', 'e')
