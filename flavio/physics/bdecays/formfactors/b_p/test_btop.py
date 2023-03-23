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

    def test_bcl_lmvd(self):
        # compare to results obtained from EOS (see https://gist.github.com/peterstangl/7d6c862bff87a10e7334993acd2ae0c5 for the notebook used)
        # see also fig. 5 in arXiv:2102.07233
        q2vals = [-10, -5, 0, 5, 10, 15, 20, 25]
        eos_Btopi = np.array([[0.13944073, 0.1786386 , 0.23467602, 0.31969463,
                               0.46026432, 0.7264543 , 1.37263015, 4.34113315],
                              [0.19729307, 0.21337439, 0.23467602, 0.26452324,
                               0.30960826, 0.38517962, 0.53237519, 0.87948273],
                              [0.14937763, 0.18405398, 0.23500764, 0.31456728,
                               0.44987482, 0.71224394, 1.35505172, 4.19770336]])

        for i, q2 in enumerate(q2vals):
            ff_bcl_lmvd = Implementation['B->pi BCL4-LMVD'].get_central(constraints_obj=c, wc_obj=None, q2=q2)
            self.assertAlmostEqual(ff_bcl_lmvd['f+'], eos_Btopi[0,i], places=5, msg="Failed for f+ in B->pi at q2={}".format(q2))
            self.assertAlmostEqual(ff_bcl_lmvd['f0'], eos_Btopi[1,i], places=5, msg="Failed for f0 in B->pi at q2={}".format(q2))
            self.assertAlmostEqual(ff_bcl_lmvd['fT'], eos_Btopi[2,i], places=5, msg="Failed for fT in B->pi at q2={}".format(q2))
