import unittest
import numpy as np
from . import wilsoncoefficients
from .. import eft
import flavio

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
    'm_s': 0.1,
    'm_t': 173.1,
    'm_c': 1.275,
    'GF': 1.1663787e-5,
    'Vus': 0.22,
    'Vub': 3.7e-3,
    'Vcb': 4.1e-2,
    'delta': 1.22,
}

class TestBWilson(unittest.TestCase):
    def test_wctot(self):
        wc_low_correct = np.array([ -2.93671059e-01,   1.01676402e+00,  -5.87762813e-03,
        -8.70666812e-02,   4.11098919e-04,   1.10641294e-03,
        -2.95662859e-01,  -1.63048361e-01,   4.11363023e+00,
        -4.19345312e+00,   3.43507549e-03,   1.22202095e-03,
        -1.03192325e-03,  -1.00703396e-04,  -3.17810374e-03])
        wc_obj = eft.WilsonCoefficients()
        wc_low = wilsoncoefficients.wctot_dict(wc_obj, 'bsmumu', 4.2, par)
        wc_names = ['C1_bs', 'C2_bs', 'C3_bs', 'C4_bs', 'C5_bs', 'C6_bs', 'C7_bs', 'C8_bs', 'C9_bsmumu', 'C10_bsmumu', 'C3Q_bs', 'C4Q_bs', 'C5Q_bs', 'C6Q_bs', 'Cb_bs', 'C1p_bs', 'C2p_bs', 'C3p_bs', 'C4p_bs', 'C5p_bs', 'C6p_bs', 'C7p_bs', 'C8p_bs', 'C9p_bsmumu', 'C10p_bsmumu', 'C3Qp_bs', 'C4Qp_bs', 'C5Qp_bs', 'C6Qp_bs', 'Cbp_bs', 'CS_bsmumu', 'CP_bsmumu', 'CSp_bsmumu', 'CPp_bsmumu']
        wc_low_array = np.asarray([wc_low[key] for key in wc_names])
        yi = np.array([0, 0, -1/3., -4/9., -20/3., -80/9.])
        zi = np.array([0, 0, 1, -1/6., 20, -10/3.])
        wc_low_array[6] = wc_low_array[6] + np.dot(yi, wc_low_array[:6]) # c7eff
        wc_low_array[7] = wc_low_array[7] + np.dot(zi, wc_low_array[:6]) # c8eff
        np.testing.assert_almost_equal(wc_low_array[:15], wc_low_correct, decimal=2)

    def test_C78p(self):
        wc_obj = eft.WilsonCoefficients()
        wc_low = wilsoncoefficients.wctot_dict(wc_obj, 'bsmumu', 4.2, par)
        ms = flavio.physics.running.running.get_ms(par, 4.2, nf_out=5)
        mb = flavio.physics.running.running.get_mb(par, 4.2, nf_out=5)
        self.assertAlmostEqual(wc_low['C7p_bs']/wc_low['C7_bs'], ms/mb)
        self.assertAlmostEqual(wc_low['C8p_bs']/wc_low['C8_bs'], ms/mb)

    def test_clnu(self):
        par_dict = flavio.default_parameters.get_central_all()
        par_dict['alpha_s'] = 0.1184
        par_dict['alpha_e'] = 1/127.925
        par_dict['s2w']  = 0.2315
        par_dict['m_t']  = 173.3
        cl = wilsoncoefficients.CL_SM(par_dict)
        # comparing the central value of X_t to (4.2) of 1009.0947
        self.assertAlmostEqual(-cl*par_dict['s2w']/1.469, 1, delta=0.01)
