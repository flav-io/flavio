import unittest
import numpy as np
from . import amplitude, rge, observables
from math import sin, asin
from flavio.physics.eft import WilsonCoefficients
from flavio.physics.mesonmixing.common import wcnp_dict

s = 1.519267515435317e+24

par = {
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
}

wc_obj = WilsonCoefficients()
wc_B0 = wcnp_dict(wc_obj, 'df2_bd', 4.2, par)
wc_Bs = wcnp_dict(wc_obj, 'df2_bs', 4.2, par)
wc_K = wcnp_dict(wc_obj, 'df2_sd', 2, par)

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
        # check whether order of magnitudes of SM predictions are righ
        self.assertAlmostEqual(observables.DeltaM(wc_obj, par, 'B0')*1e-12*s, 0.53, places=1)
        self.assertAlmostEqual(observables.DeltaM(wc_obj, par, 'Bs')*1e-12*s, 18, places=-1)
        # self.assertAlmostEqual(sin(observables.phi(wc_obj, par, 'B0')), 0.73, places=2)
        # self.assertAlmostEqual(observables.phi(wc_obj, par, 'Bs'), -0.038, places=3)
        self.assertAlmostEqual(observables.S_BJpsiK(wc_obj, par), 0.73, places=2)
        self.assertAlmostEqual(observables.S_Bspsiphi(wc_obj, par), asin(-0.038), places=3)


    def test_running(self):
        c_in = np.array([ 0.20910694,  0.77740198,  0.54696337,  0.46407456,  0.42482153,
        0.95717777,  0.62733321,  0.87053086])
        c_out = rge.run_wc_df2(par, c_in, 173.3, 2)
        c_out_U = np.dot(U_2GeV, c_in)
        # FIXME this should work better
        np.testing.assert_almost_equal(c_out/c_out_U, np.ones(8), decimal=1)
        # compare eta at 2 GeV to the values in table 2 of hep-ph/0102316
        par_bju = par.copy()
        par_bju['alpha_s'] = 0.118
        par_bju[('mass','b')] = 4.4
        c_out_bju = rge.run_wc_df2(par_bju, c_in, 166., 2)
        self.assertAlmostEqual(c_out_bju[0]/c_in[0], 0.788, places=2)
