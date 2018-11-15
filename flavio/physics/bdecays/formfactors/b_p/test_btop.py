import unittest
from math import sqrt,radians,asin
from flavio.physics.bdecays.formfactors.b_p import btop, bcl_parameters, bcl, bsz_parameters
from flavio.physics.bdecays.formfactors.b_v.test_btov import test_eos_ff
import numpy as np
import copy
from flavio.parameters import default_parameters
from flavio.classes import Parameter, Implementation

c = copy.deepcopy(default_parameters)

class TestBtoP(unittest.TestCase):
    def test_bcl_iw(self):
        c = copy.deepcopy(default_parameters)
        par = c.get_central_all()
        bcl.ff_isgurwise('B->D', 1, par, 4.8, n=3)
        Implementation['B->D BCL3-IW'].get_central(constraints_obj=c, wc_obj=None, q2=1)
        # assert f+(0)=f0(0)
        ffq20 = Implementation['B->D BCL3-IW'].get_central(constraints_obj=c, wc_obj=None, q2=0)
        self.assertEqual(ffq20['f+'], ffq20['f0'])

    def test_lattice(self):
        # compare to digitized fig. 18 of 1509.06235v1
        ff_latt = Implementation['B->K BCL3'].get_central(constraints_obj=c, wc_obj=None, q2=0)
        self.assertAlmostEqual(ff_latt['f+'], 0.33, places=1)
        self.assertAlmostEqual(ff_latt['f0'], 0.33, places=1)
        self.assertAlmostEqual(ff_latt['fT'], 0.26, places=1)
        ff_latt = Implementation['B->K BCL3'].get_central(constraints_obj=c, wc_obj=None, q2=5)
        self.assertAlmostEqual(ff_latt['f+'], 0.42, places=1)
        self.assertAlmostEqual(ff_latt['f0'], 0.36, places=1)
        self.assertAlmostEqual(ff_latt['fT'], 0.37, places=1)
        ff_latt = Implementation['B->K BCL3'].get_central(constraints_obj=c, wc_obj=None, q2=10)
        self.assertAlmostEqual(ff_latt['f+'], 0.64, places=1)
        self.assertAlmostEqual(ff_latt['f0'], 0.44, places=1)
        self.assertAlmostEqual(ff_latt['fT'], 0.57, places=1)
        ff_latt = Implementation['B->K BCL3'].get_central(constraints_obj=c, wc_obj=None, q2=15)
        self.assertAlmostEqual(ff_latt['f+'], 0.98, places=1)
        self.assertAlmostEqual(ff_latt['f0'], 0.56, places=1)
        self.assertAlmostEqual(ff_latt['fT'], 0.96, places=1)
        ff_latt = Implementation['B->K BCL3'].get_central(constraints_obj=c, wc_obj=None, q2=20)
        self.assertAlmostEqual(ff_latt['f+'], 1.7, places=1)
        self.assertAlmostEqual(ff_latt['f0'], 0.71, places=1)
        self.assertAlmostEqual(ff_latt['fT'], 1.74, places=1)

    def test_bsz(self):
        bsz_parameters.load_parameters('data/arXiv-1811-00983v1/BD_LCSR.json', 'B->D', c)
        ff_latt = Implementation['B->D BSZ3'].get_central(constraints_obj=c, wc_obj=None, q2=0)

    def test_gkvd(self):
        # compare to numbers of arXiv:1811.00983
        c = copy.deepcopy(default_parameters)
        for q2 in [1.5, 6]:
            for ff in ['f+', 'f0']:
                for P in ['K', 'D', 'pi']:
                    bsz_parameters.gkvd_load('v1', 'LCSR-Lattice', ('B->{}'.format(P),), c)
                    ffbsz3 = Implementation['B->{} BSZ3'.format(P)].get_central(constraints_obj=c, wc_obj=None, q2=q2)
                    self.assertAlmostEqual(ffbsz3[ff] / test_eos_ff[P][q2][ff],
                                           1,
                                           places=3,
                                           msg="Failed for {} in B->{} at q2={}".format(ff, P, q2))
