import unittest
from .bll import *
import numpy as np
from .. import ckm
from math import radians

s = 1.519267515435317e+24

# parameters taken from PDG and table I of 1311.0903
par = {
    ('mass','e'): 0.510998928e-3,
    ('mass','mu'): 105.6583715e-3,
    ('mass','tau'): 1.77686,
    ('mass','Bs'): 5.36677,
    ('mass','b'): 4.17,
    ('mass','t'): 173.21,
    ('mass','c'): 1.275,
    ('lifetime','Bs'): 1.516e-12*s,
    ('lifetime','Bd'): 1.519e-12*s,
    'Gmu': 1.166379e-5,
    'alpha_e': 1/127.944,
    'alpha_s': 0.1184,
    ('mass','Z'): 91.1876,
    ('f','Bs'): 0.2277,
    ('f','Bd'): 0.1905,
    'Vus': 0.2254,
    'Vcb': 4.24e-2,
    'Vub': 3.82e-3,
    'gamma': radians(73.),
    ('DeltaGamma/Gamma','Bs'): 0.1226,
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
        # correction factor should enhance the BR by roughly 7%
        self.assertAlmostEqual(br_timeint(par, wc, 'Bs', 'mu')/br_inst(par, wc, 'Bs', 'mu'), 1.07, places=2)
        # ratio of Bs->mumu and Bs->ee BRs should be roughly given by ratio of squared masses
        self.assertAlmostEqual(
            br_timeint(par, wc, 'Bs', 'e')/br_timeint(par, wc, 'Bs', 'mu')/par[('mass','e')]**2*par[('mass','mu')]**2,
            1., places=2)
        # comparison to 1311.0903
        self.assertAlmostEqual(abs(ckm.xi('t','bs')(par))/par['Vcb'], 0.980, places=3)
        self.assertAlmostEqual(br_timeint(par, wc, 'Bs', 'mu')/3.65e-9, 1, places=1)
        self.assertAlmostEqual(br_timeint(par, wc, 'Bs', 'e')/8.54e-14, 1, places=1)
        self.assertAlmostEqual(br_timeint(par, wc, 'Bs', 'tau')/7.73e-7, 1, places=1)
