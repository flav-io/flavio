import unittest
import numpy as np
from flavio.physics.bdecays import bpll_new, bpll
from flavio.physics.eft import WilsonCoefficients
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict

s = 1.519267515435317e+24

par = {
    ('mass','e'): 0.510998928e-3,
    ('mass','mu'): 105.6583715e-3,
    ('mass','tau'): 1.77686,
    ('mass','B+'): 5.27929,
    ('mass','B0'): 5.27961,
    ('mass','K0'): 0.497611,
    ('mass','K+'): 0.493677,
    ('lifetime','B+'): 1638.e-15*s,
    'Gmu': 1.1663787e-5,
    'alpha_e': 1/127.940,
    'alpha_s': 0.1185,
    ('mass','Z'): 91.1876,
    ('mass','b'): 4.17,
    ('mass','t'): 173.21,
    ('mass','c'): 1.275,
    'Vus': 0.22,
    'Vub': 3.7e-3,
    'Vcb': 4.1e-2,
    'gamma': 1.22,
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

wc_obj = WilsonCoefficients()
wc_ini = {
'CS_bsmumu': 1.31,
'CP_bsmumu': -3.25,
'CSp_bsmumu': 1.86j + 3.,
'CPp_bsmumu': -1.38j,
'C7effp_bs': -3.25,
'C9p_bsmumu': 1.86j + 3.,
'C10p_bsmumu': -1.38j,
}
wc = wctot_dict(wc_obj, 'bsmumu', 4.8, par)

class TestBPll(unittest.TestCase):
    def test_bkll(self):
        # just some trivial tests to see if calling the functions raises an error
        q2 = 2.
        a = bpll.amps(q2, wc, par, 'B+', 'K+', 'mu')
        ac_old = bpll.angulardist(a, q2, par, 'B+', 'K+', 'mu')
        ac_new = bpll_new.get_angularcoeff(q2, wc_obj, par, 'B+', 'K+', 'mu')
        for k in ac_old.keys():
            if ac_new[k] != 0:
                self.assertAlmostEqual(ac_old[k]/(ac_new[k]), 1, places=12)
