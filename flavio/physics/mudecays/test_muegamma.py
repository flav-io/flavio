import unittest
import flavio
import numpy as np
from cmath import exp, pi
from wilson import Wilson

from flavio.physics.taudecays.test_taulgamma import compare_BR


class TestMuEGamma(unittest.TestCase):
    def test_muegamma(self):
        self.assertEqual(flavio.sm_prediction('BR(mu->egamma)'), 0)
    def test_tauegamma_implementation(self):
        input_dict_list=[{
        'Cgamma_mue':np.random.random()*1e-8*exp(1j*2*pi*np.random.random()),
        'Cgamma_emu':np.random.random()*1e-8*exp(1j*2*pi*np.random.random()),
        } for i in range(10)]
        BRs = np.array([
            flavio.np_prediction(
                'BR(mu->egamma)',
                Wilson(input_dict, 100,  'WET', 'flavio')
            )
            for input_dict in input_dict_list
        ])
        compare_BRs = np.array([
            compare_BR(
                Wilson(input_dict, 100,  'WET', 'flavio'),
                'mu', 'e',
            )
            for input_dict in input_dict_list
        ])
        self.assertAlmostEqual(np.max(np.abs(1-BRs/compare_BRs)), 0, delta=0.005)
