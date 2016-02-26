import unittest
from math import sqrt,radians,asin
from flavio.physics.bdecays.formfactors.b_v import btov, bsz_parameters, lattice_parameters, cln
import numpy as np
from flavio.classes import Implementation
from flavio.parameters import default_parameters
import copy


class TestBtoV(unittest.TestCase):

    def test_cln(self):
        c = copy.deepcopy(default_parameters)
        par = c.get_central_all()
        cln.ff('B->D*', 1, par, 4.8)
        Implementation.get_instance('B->D* CLN-IW').get_central(constraints_obj=c, wc_obj=None, q2=1)

    def test_bsz3(self):
        c = copy.deepcopy(default_parameters)
        bsz_parameters.bsz_load_v1_lcsr(c)
        # compare to numbers in table 4 of arXiv:1503.05534v1
        # B->K* all FFs
        ffbsz3 = Implementation.get_instance('B->K* BSZ3').get_central(constraints_obj=c, wc_obj=None, q2=0)
        self.assertAlmostEqual(ffbsz3['A0'], 0.391, places=2)
        self.assertAlmostEqual(ffbsz3['A1'], 0.289, places=3)
        self.assertAlmostEqual(ffbsz3['A12'], 0.281, places=3)
        self.assertAlmostEqual(ffbsz3['V'], 0.366, places=3)
        self.assertAlmostEqual(ffbsz3['T1'], 0.308, places=3)
        self.assertAlmostEqual(ffbsz3['T23'], 0.793, places=3)
        self.assertAlmostEqual(ffbsz3['T1'], ffbsz3['T2'], places=16)
        # A1 for the remaining transitions
        ffbsz3 = Implementation.get_instance('B->rho BSZ3').get_central(constraints_obj=c, wc_obj=None, q2=0)
        self.assertAlmostEqual(ffbsz3['A1'], 0.267, places=3)
        ffbsz3 = Implementation.get_instance('B->omega BSZ3').get_central(constraints_obj=c, wc_obj=None, q2=0)
        self.assertAlmostEqual(ffbsz3['A1'], 0.237, places=3)
        ffbsz3 = Implementation.get_instance('Bs->phi BSZ3').get_central(constraints_obj=c, wc_obj=None, q2=0)
        self.assertAlmostEqual(ffbsz3['A1'], 0.315, places=3)
        ffbsz3 = Implementation.get_instance('Bs->K* BSZ3').get_central(constraints_obj=c, wc_obj=None, q2=0)
        self.assertAlmostEqual(ffbsz3['A1'], 0.246, places=3)
    #
    def test_lattice(self):
        c = copy.deepcopy(default_parameters)
        lattice_parameters.lattice_load(c)
        fflatt = Implementation.get_instance('B->K* SSE').get_central(constraints_obj=c, wc_obj=None, q2=12.)
        self.assertAlmostEqual(fflatt['V'], 0.84, places=2)
        self.assertAlmostEqual(fflatt['A0'], 0.861, places=3)
        self.assertAlmostEqual(fflatt['A1'], 0.440, places=3)
        self.assertAlmostEqual(fflatt['A12'], 0.339, places=3)
        self.assertAlmostEqual(fflatt['T1'], 0.711, places=3)
        self.assertAlmostEqual(fflatt['T2'], 0.433, places=3)
        self.assertAlmostEqual(fflatt['T23'], 0.809, places=3)
        fflatt = Implementation.get_instance('Bs->phi SSE').get_central(constraints_obj=c, wc_obj=None, q2=12.)
        self.assertAlmostEqual(fflatt['V'], 0.767, places=2)
        self.assertAlmostEqual(fflatt['A0'], 0.907, places=2)
        self.assertAlmostEqual(fflatt['A1'], 0.439, places=2)
        self.assertAlmostEqual(fflatt['A12'], 0.321, places=2)
        self.assertAlmostEqual(fflatt['T1'], 0.680, places=2)
        self.assertAlmostEqual(fflatt['T2'], 0.439, places=2)
        self.assertAlmostEqual(fflatt['T23'], 0.810, places=2)
        fflatt = Implementation.get_instance('Bs->K* SSE').get_central(constraints_obj=c, wc_obj=None, q2=12.)
        self.assertAlmostEqual(fflatt['V'], 0.584, places=3)
        self.assertAlmostEqual(fflatt['A0'], 0.884, places=3)
        self.assertAlmostEqual(fflatt['A1'], 0.370, places=3)
        self.assertAlmostEqual(fflatt['A12'], 0.321, places=3)
        self.assertAlmostEqual(fflatt['T1'], 0.605, places=3)
        self.assertAlmostEqual(fflatt['T2'], 0.383, places=3)
        self.assertAlmostEqual(fflatt['T23'], 0.743, places=3)
