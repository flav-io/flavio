import unittest
from math import sqrt,radians,asin
from flavio.physics.bdecays.formfactors.b_p import btop
import numpy as np
import copy
from flavio.parameters import default_parameters
from flavio.classes import Parameter, Implementation

c = copy.copy(default_parameters)
Parameter('B->K BCL a0_f+')
Parameter('B->K BCL a0_f0')
Parameter('B->K BCL a0_fT')
Parameter('B->K BCL a1_f+')
Parameter('B->K BCL a1_f0')
Parameter('B->K BCL a1_fT')
Parameter('B->K BCL a2_f+')
Parameter('B->K BCL a2_f0')
Parameter('B->K BCL a2_fT')
c.set_constraint('B->K BCL a0_f+', 0.466)
c.set_constraint('B->K BCL a1_f+', -0.885)
c.set_constraint('B->K BCL a2_f+', -0.213)
c.set_constraint('B->K BCL a0_f0', 0.292)
c.set_constraint('B->K BCL a1_f0', 0.281)
c.set_constraint('B->K BCL a2_f0', 0.150)
c.set_constraint('B->K BCL a0_fT', 0.460)
c.set_constraint('B->K BCL a1_fT', -1.089)
c.set_constraint('B->K BCL a2_fT', -1.114)

class TestBtoP(unittest.TestCase):
    def test_lattice(self):
        # compare to digitized fig. 18 of 1509.06235v1
        ff_latt = Implementation.get_instance('B->K BCL').get_central(constraints_obj=c, wc_obj=None, q2=0)
        self.assertAlmostEqual(ff_latt['f+'], 0.33, places=1)
        self.assertAlmostEqual(ff_latt['f0'], 0.33, places=1)
        self.assertAlmostEqual(ff_latt['fT'], 0.26, places=1)
        ff_latt = Implementation.get_instance('B->K BCL').get_central(constraints_obj=c, wc_obj=None, q2=5)
        self.assertAlmostEqual(ff_latt['f+'], 0.42, places=1)
        self.assertAlmostEqual(ff_latt['f0'], 0.36, places=1)
        self.assertAlmostEqual(ff_latt['fT'], 0.37, places=1)
        ff_latt = Implementation.get_instance('B->K BCL').get_central(constraints_obj=c, wc_obj=None, q2=10)
        self.assertAlmostEqual(ff_latt['f+'], 0.64, places=1)
        self.assertAlmostEqual(ff_latt['f0'], 0.44, places=1)
        self.assertAlmostEqual(ff_latt['fT'], 0.57, places=1)
        ff_latt = Implementation.get_instance('B->K BCL').get_central(constraints_obj=c, wc_obj=None, q2=15)
        self.assertAlmostEqual(ff_latt['f+'], 0.98, places=1)
        self.assertAlmostEqual(ff_latt['f0'], 0.56, places=1)
        self.assertAlmostEqual(ff_latt['fT'], 0.96, places=1)
        ff_latt = Implementation.get_instance('B->K BCL').get_central(constraints_obj=c, wc_obj=None, q2=20)
        self.assertAlmostEqual(ff_latt['f+'], 1.7, places=1)
        self.assertAlmostEqual(ff_latt['f0'], 0.71, places=1)
        self.assertAlmostEqual(ff_latt['fT'], 1.74, places=1)
