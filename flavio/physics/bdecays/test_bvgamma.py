import unittest
import numpy as np
from .bvgamma import *
from flavio.physics.bdecays.formfactors.b_v import bsz_parameters


s = 1.519267515435317e+24

par = {
    ('mass','B+'): 5.27929,
    ('mass','B0'): 5.27961,
    ('mass','Bs'): 5.36679,
    ('mass','K*0'): 0.89581,
    ('mass','K*+'): 0.89166,
    ('lifetime','B+'): 1638.e-15*s,
    ('lifetime','B0'): 152.e-14*s,
    'alpha_e': 1/127.940,
    'alpha_s': 0.1185,
    ('mass','Z'): 91.1876,
    ('mass','b'): 4.17,
    'Gmu': 1.1663787e-5,
    'Vus': 0.22,
    'Vub': 3.7e-3,
    'Vcb': 4.1e-2,
    'gamma': 1.22,
}

par.update(bsz_parameters.ffpar_lcsr)

wc = {
    'C7': 0,
    'C7p': 0,
    'C9': 0,
    'C9p': 0,
    'C10': 0,
    'C10p': 0,
    'CP': 0,
    'CPp': 0,
    'CS': 0,
    'CSp': 0,
}

class TestBVgamma(unittest.TestCase):
    def test_bksgamma(self):
        # just some trivial tests to see if calling the functions raises an error
        prefactor(par, 'B0', 'K*0')
        a = amps(wc, par, 'B0', 'K*0')
        S(wc, par, 'B0', 'K*0')
        # ACP should vanish as CP conjugation is ignored here (a=abar)
        self.assertEqual(ACP(wc, par, 'B0', 'K*0'),   0.)
        # # rough numerical comparison of CP-averaged observables to 1503.05534v1
        # # FIXME this should work much better with NLO corrections ...
        self.assertAlmostEqual(BR(wc, par, 'B0', 'K*0')*1e4/4.22, 1, places=-1)
        self.assertAlmostEqual(BR(wc, par, 'B+', 'K*+')*1e4/4.42, 1, places=-1)
