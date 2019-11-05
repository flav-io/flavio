import unittest
import numpy as np
from .eft import *
import pkgutil
from wilson import wcxf
import wilson

par = {
    'm_Z': 91.1876,
    'm_b': 4.18,
    'm_d': 4.8e-3,
    'm_s': 0.095,
    'm_t': 173.3,
    'm_c': 1.27,
    'm_u': 2.3e-3,
    'alpha_e': 1/127.944,
    'alpha_s': 0.1185,
    'GF': 1.1663787e-5,
}

class TestEFT(unittest.TestCase):
    def test_eft_old(self):
        wc =  WilsonCoefficients()
        wc.set_initial({'CVLL_bsbs': 0.1j, 'C9_bsmumu':-1.5, 'CVL_bctaunutau': 0.2}, 160.)
        d1 = wc.get_wc('bsbs', 4.8, par)
        d2 = wc.get_wc('bsbs', 4.8, par)  # again, to test the cache
        self.assertDictEqual(d1, d2)
        wc.get_wc('bsmumu', 4.8, par)
        wc.get_wc('bctaunutau', 4.8, par)

    def test_set_initial_wcxf(self):
        test_file = pkgutil.get_data('flavio', 'data/test/wcxf-flavio-example.yml')
        flavio_wc = WilsonCoefficients()
        wcxf_wc = wcxf.WC.load(test_file.decode('utf-8'))
        wcxf_wc.validate()
        flavio_wc.set_initial_wcxf(wcxf_wc)
        wc_out = flavio_wc.get_wc('bsee', 160, par)
        self.assertEqual(wc_out['C9_bsee'], -1+0.01j)
        self.assertEqual(wc_out['C9p_bsee'], 0.1)
        self.assertEqual(wc_out['C10_bsee'], 0.05j)
        self.assertEqual(wc_out['C10p_bsee'], 0.1-0.3j)
        self.assertEqual(wc_out['CS_bsee'], 0)
        wcxf_wc.basis = 'unknown basis'
        with self.assertRaises((KeyError, ValueError, AssertionError)):
            flavio_wc.set_initial_wcxf(wcxf_wc)

    def test_set_initial_wcxf_minimal(self):
        for eft in ['WET', 'WET-4', 'WET-3']:
            wc = wcxf.WC(eft, 'flavio', 120, {'CVLL_sdsd': {'Im': 1}})
            fwc = WilsonCoefficients()
            fwc.set_initial_wcxf(wc)
            self.assertEqual(fwc.get_wc('sdsd', 120, par, eft=eft)['CVLL_sdsd'], 1j)
            pf = 4 * par['GF'] / np.sqrt(2)
            wc = wcxf.WC(eft, 'Bern', 120, {'1dsds': {'Im': 1/pf}})
            fwc = WilsonCoefficients()
            fwc.set_initial_wcxf(wc)
            self.assertAlmostEqual(fwc.get_wc('sdsd', 120, par, eft=eft)['CVLL_sdsd'], 1j)

    def tets_repr(self):
        wc = WilsonCoefficients()
        wc._repr_markdown_()
        wc.set_initial({'C7_bs': -0.1}, 5)
        wc._repr_markdown_()

    def test_get_initial_wcxf_minimal(self):
        for eft in ['WET', 'WET-4', 'WET-3']:
            wc = wcxf.WC(eft, 'flavio', 120, {'CVLL_sdsd': {'Im': 1}})
            fwc = WilsonCoefficients()
            fwc.set_initial_wcxf(wc)
            wc2 = fwc.get_initial_wcxf
            self.assertEqual(wc.eft, wc2.eft)
            self.assertEqual(wc2.basis, 'flavio')
            self.assertDictEqual(wc.dict, wc2.dict)


    def test_deprecations(self):
        """Check that deprecated or renamed Wilson coefficients raise/warn"""
        wc = WilsonCoefficients()
        wc.set_initial({'C9_bsmumu': 1.2}, 5)  # this should work
        with self.assertRaises((KeyError, AssertionError)):
            wc.set_initial({'C9_bsmumu': 1.2, 'C7effp_bs': 3}, 5)
        with self.assertRaises((KeyError, AssertionError)):
            wc.set_initial({'C9_bsmumu': 1.2, 'C8eff_sd': 3}, 5)
        with self.assertRaises((KeyError, AssertionError)):
            wc.set_initial({'C9_bsmumu': 1.2, 'CV_bcenu': 3}, 5)
        with self.assertRaises((KeyError, AssertionError)):
            wc.set_initial({'C3Qp_bs': 1.2, 'C1_bs': 3}, 5)
