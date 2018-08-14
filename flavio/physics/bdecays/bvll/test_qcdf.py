import unittest
import numpy as np
from flavio.physics.bdecays.bvll.qcdf import *
from flavio.physics.bdecays.bvll.amplitudes import *
from flavio.physics.eft import WilsonCoefficients
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict
from flavio.physics.bdecays.formfactors.b_v import bsz_parameters
from flavio.physics.running import running
from flavio.parameters import default_parameters
import copy

s = 1.519267515435317e+24

c = copy.deepcopy(default_parameters)
bsz_parameters.bsz_load_v1_lcsr(c)
par = c.get_central_all()

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
        # (commented lines correspond to errors in the old code)
        np.testing.assert_almost_equal(L1(1.), 0, decimal=5)
        np.testing.assert_almost_equal(L1(-1.), -0.582241, decimal=5)
        np.testing.assert_almost_equal(L1(0), -1.64493, decimal=5)
        np.testing.assert_almost_equal(L1(1.+1.j), 0.205617-0.915966j, decimal=5)
        np.testing.assert_almost_equal(L1(1.-1.j), 0.205617+0.915966j, decimal=5)
        np.testing.assert_almost_equal(i1_bfs(q2, u, 4.8, 5.27961), -0.0768, decimal=3)
        # np.testing.assert_almost_equal(i1_bfs(q2, u, 1.685, 5.27961), -0.659-2.502j, decimal=3)
        np.testing.assert_almost_equal(B0diffBFS(q2, u, 4.8, 5.27961)/(0.09634), 1, decimal=3)
        # np.testing.assert_almost_equal(B0diffBFS(q2, u, 1.685, 5.27961)/(2.377-1.6506j), 1, decimal=3)
        np.testing.assert_almost_equal(t_perp(q2, u, 4.8, par, B, V), -0.442057, decimal=2)
        # np.testing.assert_almost_equal(t_perp(q2, u, 1.685, par, B, V)/(-0.004-26.1188j), 1, decimal=2)
        np.testing.assert_almost_equal(t_perp(q2, u, 0, par, B, V)/5.1578, 1, decimal=2)
        np.testing.assert_almost_equal(T_para_minus_WA(q2, par, wc, B, V, scale)/-0.124511, 1, decimal=0)
        np.testing.assert_almost_equal(T_para_minus_O8(q2, par, wc, B, V, u, scale)/-4.7384/-0.16718/0.023037, 1, decimal=1)
        np.testing.assert_almost_equal(T_para_minus_QSS(q2, par, wc, B, V, u, scale)/(-2.045-1.608j)/0.023037, 1, decimal=1)
        np.testing.assert_almost_equal(T_perp_plus_O8(q2, par, wc, B, V, u, scale)/2.369/-0.1672/0.023037, 1, decimal=1)
        # np.testing.assert_almost_equal(T_perp_plus_QSS(q2, par, wc, B, V, u, scale)/(-0.0136-10.1895j)/0.023037, 1, decimal=1)
        # np.testing.assert_almost_equal(T_perp_plus_QSS(q2, par, wc, B, V, u, scale)/(0.1997-10.153j)/0.023037, 1, decimal=1)
        q2 = 1.
        # np.testing.assert_almost_equal(T_perp(q2, par, wc, B, V, scale)/(-0.001556-0.00835j), 1,  decimal=0)
