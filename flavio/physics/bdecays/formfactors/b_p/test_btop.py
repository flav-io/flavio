import unittest
from math import sqrt,radians,asin
from flavio.physics.bdecays.formfactors.b_p import btop
import numpy as np

par = {
    ('mass','B0'): 5.27961,
    ('mass','Bs'): 5.36679,
    ('mass','K0'): 0.497611,
# table XII of 1509.06235v1
    ('formfactor','B->K','a0_f+'): 0.466,
    ('formfactor','B->K','a1_f+'): -0.885,
    ('formfactor','B->K','a2_f+'): -0.213,
    ('formfactor','B->K','a0_f0'): 0.292,
    ('formfactor','B->K','a1_f0'): 0.281,
    ('formfactor','B->K','a2_f0'): 0.150,
    ('formfactor','B->K','a0_fT'): 0.460,
    ('formfactor','B->K','a1_fT'): -1.089,
    ('formfactor','B->K','a2_fT'): -1.114,
}

class TestBtoP(unittest.TestCase):
    def test_lattice(self):
        # compare to digitized fig. 18 of 1509.06235v1
        ff_latt = btop.lattice.get_ff('B->K', 0, par)
        self.assertAlmostEqual(ff_latt['f+'], 0.33, places=1)
        self.assertAlmostEqual(ff_latt['f0'], 0.33, places=1)
        self.assertAlmostEqual(ff_latt['fT'], 0.26, places=1)
        ff_latt = btop.lattice.get_ff('B->K', 5, par)
        self.assertAlmostEqual(ff_latt['f+'], 0.42, places=1)
        self.assertAlmostEqual(ff_latt['f0'], 0.36, places=1)
        self.assertAlmostEqual(ff_latt['fT'], 0.37, places=1)
        ff_latt = btop.lattice.get_ff('B->K', 10, par)
        self.assertAlmostEqual(ff_latt['f+'], 0.64, places=1)
        self.assertAlmostEqual(ff_latt['f0'], 0.44, places=1)
        self.assertAlmostEqual(ff_latt['fT'], 0.57, places=1)
        ff_latt = btop.lattice.get_ff('B->K', 15, par)
        self.assertAlmostEqual(ff_latt['f+'], 0.98, places=1)
        self.assertAlmostEqual(ff_latt['f0'], 0.56, places=1)
        self.assertAlmostEqual(ff_latt['fT'], 0.96, places=1)
        ff_latt = btop.lattice.get_ff('B->K', 20, par)
        self.assertAlmostEqual(ff_latt['f+'], 1.7, places=1)
        self.assertAlmostEqual(ff_latt['f0'], 0.71, places=1)
        self.assertAlmostEqual(ff_latt['fT'], 1.74, places=1)
