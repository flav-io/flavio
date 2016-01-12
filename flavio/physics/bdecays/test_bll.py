import unittest
from bll import *
import numpy as np

s = 1.519267515435317e+24

par = {
    ('mass','e'): 0.510998928e-3,
    ('mass','mu'): 105.6583715e-3,
    ('mass','tau'): 1.77686,
    ('mass','Bs'): 5.36679,
    ('lifetime','Bs'): 1.511e-12*s,
    'Gmu': 1.1663787e-5,
    'alphaem': 1/127.940,
    ('f','Bs'): 0.2277,
    'Vus': 0.22,
    'Vub': 3.7e-3,
    'Vcb': 4.1e-2,
    'gamma': 1.22,
    ('DeltaGamma/Gamma','Bs'): 0.124,
}

wc = {
    'C10': 0,
    'C10p': 0,
    'CP': 0,
    'CPp': 0,
    'CS': 0,
    'CSp': 0,
}

class TestBll(unittest.TestCase):
    def test_bsll(self):
        # just some trivial tests to see if calling the functions raises an error
        self.assertGreater(br_lifetime_corr(0.08, -1), 0)
        self.assertEqual(len(amplitudes(par, wc, 'Bs', 'mu')), 2)
        # ADeltaGamma should be +1.0 in the SM
        self.assertEqual(ADeltaGamma(par, wc, 'Bs', 'mu'), 1.0)
        # BR should be around 3.5e-9
        self.assertAlmostEqual(br_inst(par, wc, 'Bs', 'mu')*1e9, 3.5, places=0)
        self.assertAlmostEqual(br_timeint(par, wc, 'Bs', 'mu')*1e9, 3.5, places=0)
        # correction factor should enhance the BR by roughly 7%
        self.assertAlmostEqual(br_timeint(par, wc, 'Bs', 'mu')/br_inst(par, wc, 'Bs', 'mu'), 1.07, places=2)
        # ratio of Bs->mumu and Bs->ee BRs should be roughly given by ratio of squared masses
        self.assertAlmostEqual(
            br_timeint(par, wc, 'Bs', 'e')/br_timeint(par, wc, 'Bs', 'mu')/par[('mass','e')]**2*par[('mass','mu')]**2,
            1., places=4)
