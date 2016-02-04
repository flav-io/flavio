import unittest
import numpy as np
from .bvgamma import *
from flavio.physics.bdecays.formfactors.b_v import bsz_parameters
from flavio.physics.eft import WilsonCoefficients
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict

s = 1.519267515435317e+24

par = {
    'm_B+': 5.27929,
    'm_B0': 5.27961,
    'm_Bs': 5.36679,
    'm_K*0': 0.89581,
    'm_K*+': 0.89166,
    'tau_B+': 1638.e-15*s,
    'tau_B0': 152.e-14*s,
    'm_B0': 5.27961,
    'm_Bs': 5.36679,
    'm_b': 4.18,
    'm_d': 4.8e-3,
    'm_s': 0.095,
    'm_t': 173.3,
    'm_c': 1.27,
    'm_u': 2.3e-3,
    'm_W': 80.4,
    'tau_B0': 152.e-14*s,
    'f_B0': 0.1905,
    'bag_B0_1': 1.27/1.517,
    'bag_B0_2': 0.72,
    'bag_B0_3': 0.88,
    'bag_B0_4': 0.95,
    'bag_B0_5': 1.47,
    'f_Bs': 0.2277,
    'bag_Bs_1': 1.33/1.517,
    'bag_Bs_2': 0.73,
    'bag_Bs_3': 0.89,
    'bag_Bs_4': 0.93,
    'bag_Bs_5': 1.57,
    'Gmu': 1.1663787e-5,
    'Vus': 0.22,
    'Vub': 3.7e-3,
    'Vcb': 4.1e-2,
    'gamma': 1.22,
    'alpha_e': 1/127.944,
    'alpha_s': 0.1185,
    'm_Z': 91.1876,
    'eta_tt_B0': 0.55,
    'eta_tt_Bs': 0.55,
    'eta_tt_K0': 0.57,
    'eta_cc_K0': 1.38,
    'eta_ct_K0': 0.47,
    'kappa_epsilon': 0.94,
    'DeltaM_K0': 52.93e-4/(1e-12*s),
    'Gamma12_Bs_c': -48.0,
    'Gamma12_Bs_a': 12.3,
    'Gamma12_B0_c': -49.5,
    'Gamma12_B0_a': 11.7,
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
