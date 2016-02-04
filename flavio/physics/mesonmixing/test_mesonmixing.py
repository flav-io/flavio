import unittest
import numpy as np
from . import amplitude, rge, observables
from math import sin, asin
from flavio.physics.eft import WilsonCoefficients

s = 1.519267515435317e+24

par = {
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

wc_obj = WilsonCoefficients()
wc_B0 = wc_obj.get_wc('bdbd', 4.2, par)
wc_Bs = wc_obj.get_wc('bsbs', 4.2, par)
wc_K = wc_obj.get_wc('sdsd', 2, par)

# this is the DeltaF=2 evolution matrix from mt to 2 GeV as obtained
# from the formulae in hep-ph/0102316
U_2GeV = np.fromstring("""7.879285920724351522e-01 0 0 0 0 0 0 0 0 0
0 7.879285920724351522e-01 0 0 0 0 0 0 0 0
0 0 9.055382596246783766e-01 0 0 0 -8.687751157652989775e-02 0 0 0
0 0 0 9.055382596246783766e-01 0 0 0 -8.687751157652989775e-02 0 0
0 0 0 0 2.053305845837953836e+00 0 0 0 2.948429540513763936e+00 0
0 0 0 0 0 2.053305845837953836e+00 0 0 0 2.948429540513763936e+00
0 0 -1.531397859765291969e+00 0 0 0 3.202996870321630496e+00 0 0 0
0 0 0 -1.531397859765291969e+00 0 0 0 3.202996870321630496e+00 0 0
0 0 0 0 -9.064850704338395931e-03 0 0 0 4.163390259182284114e-01 0
0 0 0 0 0 -9.064850704338395931e-03 0 0 0 4.163390259182284114e-01
""", sep=' ').reshape((10,10))[[0,4,8,1,5,9,2,6],:][:,[0,4,8,1,5,9,2,6]]

class TestMesonMixing(unittest.TestCase):
    def test_bmixing(self):
        # just some trivial tests to see if calling the functions raises an error
        m12d = amplitude.M12_d(par, wc_B0, 'B0')
        m12s = amplitude.M12_d(par, wc_Bs, 'Bs')
        # check whether order of magnitudes of SM predictions are right
        ps = 1e-12*s
        self.assertAlmostEqual(observables.DeltaM(wc_obj, par, 'B0')*ps, 0.53, places=1)
        self.assertAlmostEqual(observables.DeltaM(wc_obj, par, 'Bs')*ps, 18, places=-1)
        self.assertAlmostEqual(observables.DeltaGamma(wc_obj, par, 'B0')/0.00261*ps, 1, places=0)
        self.assertAlmostEqual(observables.DeltaGamma(wc_obj, par, 'Bs')/0.088*ps, 1, places=1)
        self.assertAlmostEqual(observables.a_fs(wc_obj, par, 'B0')/-4.7e-4, 1, places=0)
        self.assertAlmostEqual(observables.a_fs(wc_obj, par, 'Bs')/2.22e-5, 1, places=1)
        self.assertAlmostEqual(observables.S_BJpsiK(wc_obj, par), 0.73, places=2)
        self.assertAlmostEqual(observables.S_Bspsiphi(wc_obj, par), asin(-0.038), places=3)


    def test_running(self):
        c_in = {'CSLL_bsbs': 0.77740198,
             'CSLR_bsbs': 0.87053086,
             'CSRR_bsbs': 0.42482153,
             'CTLL_bsbs': 0.54696337,
             'CTRR_bsbs': 0.95717777,
             'CVLL_bsbs': 0.20910694,
             'CVLR_bsbs': 0.62733321,
             'CVRR_bsbs': 0.46407456}
        c_in = np.array([ 0.20910694,  0.77740198,  0.54696337,  0.46407456,  0.42482153,
        0.95717777,  0.62733321,  0.87053086])
        c_out = rge.run_wc_df2(par, c_in, 173.3, 2)
        c_out_U = np.dot(U_2GeV, c_in)
        # FIXME this should work better
        np.testing.assert_almost_equal(c_out/c_out_U, np.ones(8), decimal=1)
        # compare eta at 2 GeV to the values in table 2 of hep-ph/0102316
        par_bju = par.copy()
        par_bju['alpha_s'] = 0.118
        par_bju['m_b'] = 4.4
        c_out_bju = rge.run_wc_df2(par_bju, c_in, 166., 2)
        self.assertAlmostEqual(c_out_bju[0]/c_in[0], 0.788, places=2)
