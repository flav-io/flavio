import unittest
from math import sqrt,radians,asin
from flavio.physics.bdecays.formfactors.b_v import btov, bsz_parameters
import numpy as np

par = {
    ('mass','B0'): 5.27961,
    ('mass','Bs'): 5.36679,
    ('mass','K*0'): 0.89166,
    ('mass','rho0'): 0.077526,
    ('mass','omega'): 0.78265,
    ('mass','phi'): 1.019461,
}

class TestBtoV(unittest.TestCase):
    def test_bsz3(self):
        allpar = bsz_parameters.ffpar_lcsr
        allpar.update(par)
        # compare to numbers in table 4 of arXiv:1503.05534v1
        # B->K* all FFs
        ffbsz3 = btov.bsz3.get_ff('B->K*', 0., allpar)
        self.assertAlmostEqual(ffbsz3['A0'], 0.391, places=3)
        self.assertAlmostEqual(ffbsz3['A1'], 0.289, places=3)
        self.assertAlmostEqual(ffbsz3['A12'], 0.281, places=3)
        self.assertAlmostEqual(ffbsz3['V'], 0.366, places=3)
        self.assertAlmostEqual(ffbsz3['T1'], 0.308, places=3)
        self.assertAlmostEqual(ffbsz3['T23'], 0.793, places=3)
        self.assertAlmostEqual(ffbsz3['T1'], ffbsz3['T2'], places=16)
        # A1 for the remaining transitions
        ffbsz3 = btov.bsz3.get_ff('B->rho', 0., allpar)
        self.assertAlmostEqual(ffbsz3['A1'], 0.267, places=3)
        ffbsz3 = btov.bsz3.get_ff('B->omega', 0., allpar)
        self.assertAlmostEqual(ffbsz3['A1'], 0.237, places=3)
        ffbsz3 = btov.bsz3.get_ff('Bs->phi', 0., allpar)
        self.assertAlmostEqual(ffbsz3['A1'], 0.315, places=3)
        ffbsz3 = btov.bsz3.get_ff('Bs->K*', 0., allpar)
        self.assertAlmostEqual(ffbsz3['A1'], 0.246, places=3)
