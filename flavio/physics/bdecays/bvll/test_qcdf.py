import unittest
import numpy as np
from flavio.physics.bdecays.bvll.qcdf import *
from flavio.physics.bdecays.bvll.amplitudes import *
from flavio.physics.eft import WilsonCoefficients
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict
from flavio.physics.bdecays.formfactors.b_v import bsz_parameters
from flavio.physics.running import running
s = 1.519267515435317e+24

par = {
    'm_e': 0.510998928e-3,
    'm_mu': 105.6583715e-3,
    'm_tau': 1.77686,
    'm_B+': 5.27929,
    'm_B0': 5.27961,
    'm_Bs': 5.36679,
    'm_K*0': 0.89166,
    'tau_B+': 1638.e-15*s,
    'tau_B0': 152.e-14*s,
    'alpha_e': 1/127.940,
    'alpha_s': 0.1185,
    'm_Z': 91.1876,
    'm_b': 4.17,
    'm_t': 173.21,
    'm_c': 1.275,
    'Gmu': 1.1663787e-5,
    'Vus': 0.22,
    'Vub': 3.7e-3,
    'Vcb': 4.1e-2,
    'gamma': 1.2235,
    'f_B0': 0.1905,
    ('f_perp','K*0'): 0.163,
    ('f_para','K*0'): 0.211,
    ('a1_perp','K*0'): 0.03,
    ('a1_para','K*0'): 0.02,
    ('a2_perp','K*0'): 0.08,
    ('a2_para','K*0'): 0.08,
}

par.update(bsz_parameters.ffpar_lcsr)

wc_obj = WilsonCoefficients()
wc = wctot_dict(wc_obj, 'bsmumu', 4.2, par)

class TestQCDF(unittest.TestCase):
    def test_qcdf(self):
        q2=3.5
        B='B0'
        V='K*0'
        u=0.5
        scale=4.8
        # compare to David's old Mathematica code
        np.testing.assert_almost_equal(L1(1.), 0, decimal=5)
        np.testing.assert_almost_equal(L1(-1.), -0.582241, decimal=5)
        np.testing.assert_almost_equal(L1(0), -1.64493, decimal=5)
        np.testing.assert_almost_equal(L1(1.+1.j), 0.205617-0.915966j, decimal=5)
        np.testing.assert_almost_equal(L1(1.-1.j), 0.205617+0.915966j, decimal=5)
        np.testing.assert_almost_equal(i1_bfs(q2, u, 4.8, 5.27961), -0.0768, decimal=3)
        np.testing.assert_almost_equal(i1_bfs(q2, u, 1.685, 5.27961), -0.659-2.502j, decimal=3)
        np.testing.assert_almost_equal(B0diffBFS(q2, u, 4.8, 5.27961)/(0.09634), 1, decimal=3)
        np.testing.assert_almost_equal(B0diffBFS(q2, u, 1.685, 5.27961)/(2.377-1.6506j), 1, decimal=3)
        np.testing.assert_almost_equal(t_perp(q2, u, 4.8, par, B, V), -0.442057, decimal=2)
        np.testing.assert_almost_equal(t_para(q2, u, 4.8, par, B, V), -0.382169, decimal=3)
        np.testing.assert_almost_equal(t_para(q2, u, 1.685, par, B, V)/(1.4735-27.1448j), 1, decimal=2)
        np.testing.assert_almost_equal(t_perp(q2, u, 1.685, par, B, V)/(-0.004-26.1188j), 1, decimal=2)
        np.testing.assert_almost_equal(t_perp(q2, u, 0, par, B, V)/5.1578, 1, decimal=2)
        np.testing.assert_almost_equal(t_para(q2, u, 0, par, B, V)/4.22531, 1, decimal=2)
        np.testing.assert_almost_equal(T_para_minus_WA(q2, par, wc, B, V, scale)/-0.124511, 1, decimal=0)
        np.testing.assert_almost_equal(T_para_minus_O8(q2, par, wc, B, V, u, scale)/-4.7384/-0.16718/0.023037, 1, decimal=1)
        np.testing.assert_almost_equal(T_para_minus_QSS(q2, par, wc, B, V, u, scale)/(-2.045-1.608j)/0.023037, 1, decimal=1)
        np.testing.assert_almost_equal(T_perp_plus_O8(q2, par, wc, B, V, u, scale)/2.369/-0.1672/0.023037, 1, decimal=1)
        # np.testing.assert_almost_equal(T_perp_plus_QSS(q2, par, wc, B, V, u, scale)/(-0.0136-10.1895j)/0.023037, 1, decimal=1)
        np.testing.assert_almost_equal(T_perp_plus_QSS(q2, par, wc, B, V, u, scale)/(0.1997-10.153j)/0.023037, 1, decimal=1)
        np.testing.assert_almost_equal(T_para_plus_QSS(q2, par, wc, B, V, u, scale)/(1.13445-21.1796j)/0.023037, 1, decimal=1)
        q2 = 1.
        np.testing.assert_almost_equal(T_perp(q2, par, wc, B, V, scale)/(-0.001556-0.00835j), 1,  decimal=0)
        np.testing.assert_almost_equal(T_para(q2, par, wc, B, V, scale)/(0.002688-0.009655j), 1,  decimal=0)
