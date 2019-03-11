import unittest
import flavio
from cmath import exp, pi, sqrt
import numpy as np
from wilson import Wilson

from flavio.physics.taudecays.taulgamma import wcxf_sector_names


def compare_BR(wc_wilson, l1, l2):
    scale = flavio.config['renormalization scale'][l1+'decays']
    ll = wcxf_sector_names[l1, l2]
    par = flavio.default_parameters.get_central_all()
    wc_obj = flavio.WilsonCoefficients.from_wilson(wc_wilson, par)
    par = flavio.parameters.default_parameters.get_central_all()
    wc = wc_obj.get_wc(ll, scale, par, nf_out=4)
    alpha = flavio.physics.running.running.get_alpha_e(par, scale, nf_out=4)
    e = sqrt(4 * pi * alpha)
    ml = par['m_' + l1]
    # cf. (18) of hep-ph/0404211
    pre = 48 * pi**3 * alpha / par['GF']**2
    DL = 2 / (e * ml) * wc['Cgamma_' + l1 + l2]
    DR = 2 / (e * ml) * wc['Cgamma_' + l2 + l1].conjugate()
    if l1 == 'tau':
        BR_SL = par['BR(tau->{}nunu)'.format(l2)]
    else:
        BR_SL = 1  # BR(mu->enunu) = 1
    return pre * (abs(DL)**2 + abs(DR)**2) * BR_SL

class TestTauLGamma(unittest.TestCase):
    def test_taulgamma(self):
        # compare to the experimental values
        self.assertEqual(flavio.sm_prediction('BR(tau->mugamma)'), 0)
        self.assertEqual(flavio.sm_prediction('BR(tau->egamma)'), 0)
    def test_taumugamma_implementation(self):
        input_dict_list=[{
        'Cgamma_taumu':np.random.random()*1e-8*exp(1j*2*pi*np.random.random()),
        'Cgamma_mutau':np.random.random()*1e-8*exp(1j*2*pi*np.random.random()),
        } for i in range(10)]
        BRs = np.array([
            flavio.np_prediction(
                'BR(tau->mugamma)',
                Wilson(input_dict, 100,  'WET', 'flavio')
            )
            for input_dict in input_dict_list
        ])
        compare_BRs = np.array([
            compare_BR(
                Wilson(input_dict, 100,  'WET', 'flavio'),
                'tau', 'mu',
            )
            for input_dict in input_dict_list
        ])
        self.assertAlmostEqual(np.max(np.abs(1-BRs/compare_BRs)), 0, delta=0.02)
    def test_tauegamma_implementation(self):
        input_dict_list=[{
        'Cgamma_taue':np.random.random()*1e-8*exp(1j*2*pi*np.random.random()),
        'Cgamma_etau':np.random.random()*1e-8*exp(1j*2*pi*np.random.random()),
        } for i in range(10)]
        BRs = np.array([
            flavio.np_prediction(
                'BR(tau->egamma)',
                Wilson(input_dict, 100,  'WET', 'flavio')
            )
            for input_dict in input_dict_list
        ])
        compare_BRs = np.array([
            compare_BR(
                Wilson(input_dict, 100,  'WET', 'flavio'),
                'tau', 'e',
            )
            for input_dict in input_dict_list
        ])
        self.assertAlmostEqual(np.max(np.abs(1-BRs/compare_BRs)), 0, delta=0.002)
