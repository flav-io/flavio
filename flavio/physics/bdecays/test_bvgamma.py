import unittest
import numpy as np
from .bvgamma import *
from flavio.physics.bdecays.formfactors.b_v import bsz_parameters
from flavio.physics.eft import WilsonCoefficients
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict

s = 1.519267515435317e+24

par = {
    ('mass','B+'): 5.27929,
    ('mass','B0'): 5.27961,
    ('mass','Bs'): 5.36679,
    ('mass','K*0'): 0.89581,
    ('mass','K*+'): 0.89166,
    ('lifetime','B+'): 1638.e-15*s,
    ('lifetime','B0'): 152.e-14*s,
    ('mass','B0'): 5.27961,
    ('mass','Bs'): 5.36679,
    ('mass','b'): 4.18,
    ('mass','d'): 4.8e-3,
    ('mass','s'): 0.095,
    ('mass','t'): 173.3,
    ('mass','c'): 1.27,
    ('mass','u'): 2.3e-3,
    ('mass','W'): 80.4,
    ('lifetime','B0'): 152.e-14*s,
    ('f','B0'): 0.1905,
    ('bag','B0',1): 1.27/1.517,
    ('bag','B0',2): 0.72,
    ('bag','B0',3): 0.88,
    ('bag','B0',4): 0.95,
    ('bag','B0',5): 1.47,
    ('f','Bs'): 0.2277,
    ('bag','Bs',1): 1.33/1.517,
    ('bag','Bs',2): 0.73,
    ('bag','Bs',3): 0.89,
    ('bag','Bs',4): 0.93,
    ('bag','Bs',5): 1.57,
    'Gmu': 1.1663787e-5,
    'Vus': 0.22,
    'Vub': 3.7e-3,
    'Vcb': 4.1e-2,
    'gamma': 1.22,
    'alpha_e': 1/127.944,
    'alpha_s': 0.1185,
    ('mass','Z'): 91.1876,
    ('eta_tt', 'B0'): 0.55,
    ('eta_tt', 'Bs'): 0.55,
    ('eta_tt', 'K0'): 0.57,
    ('eta_cc', 'K0'): 1.38,
    ('eta_ct', 'K0'): 0.47,
    'kappa_epsilon': 0.94,
    ('DeltaM','K0'): 52.93e-4/(1e-12*s),
    ('Gamma12','Bs','c'): -48.0,
    ('Gamma12','Bs','a'): 12.3,
    ('Gamma12','B0','c'): -49.5,
    ('Gamma12','B0','a'): 11.7,
}

par.update(bsz_parameters.ffpar_lcsr)

wc_obj = WilsonCoefficients()
wc = wctot_dict(wc_obj, 'bsee', 4.2, par)

class TestBVgamma(unittest.TestCase):
    def test_bksgamma(self):
        # just some trivial tests to see if calling the functions raises an error
        prefactor(par, 'B0', 'K*0')
        # S and ACP should vanish as there are no strong phases or power corrections yet
        self.assertEqual(ACP(wc_obj, par, 'B0', 'K*0'),   0.)
        self.assertEqual(S(wc_obj, par, 'B0', 'K*0'),   0.)
        # # rough numerical comparison of CP-averaged observables to 1503.05534v1
        # # FIXME this should work much better with NLO corrections ...
        self.assertAlmostEqual(BR(wc_obj, par, 'B0', 'K*0')*1e4/4.22, 1, places=-1)
        self.assertAlmostEqual(BR(wc_obj, par, 'B+', 'K*+')*1e4/4.42, 1, places=-1)
