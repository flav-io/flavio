import unittest
import flavio
from flavio.physics.taudecays import taulnunu
from wilson import Wilson
from math import sqrt


par = flavio.default_parameters.get_central_all()
wc_obj = flavio.WilsonCoefficients()


class TestTaulnunu(unittest.TestCase):
    def test_taulnunu_exp(self):
        # compare to the experimental values
        # cf. eq. (95) of 1705.00929
        self.assertAlmostEqual(flavio.sm_prediction('BR(tau->mununu)') / 0.1739,
                               1 / 1.0060, delta=0.0003)
        self.assertAlmostEqual(flavio.sm_prediction('BR(tau->enunu)') / 0.1782,
                               1 / 1.0022, delta=0.0003)

    def test_taulnunu_wrongflavor(self):
        self.assertEqual(taulnunu.BR_taulnunu(wc_obj, par, 'e', 'mu', 'e'), 0)

    def test_taulnunu_np(self):
        w = Wilson({'CVLL_numunuetaue': 2e-4, 'CVLL_numunueetau': 1e-4},
                    80, 'WET', 'flavio')
        wc_np = flavio.WilsonCoefficients.from_wilson(w, par)
        BR1 = taulnunu.BR_taulnunu(wc_np, par, 'e', 'e', 'mu')
        BR2 = taulnunu.BR_taulnunu(wc_np, par, 'e', 'mu', 'e')
        self.assertEqual(BR1 / BR2, 4)

    def test_taulnunu_np2(self):
        w = Wilson({'CVLL_numunuetaue': 2e-2, 'CVLL_numunuetaumu': 1e-2},
                    80, 'WET', 'flavio')
        BR1 = flavio.np_prediction('BR(tau->enunu)', w)
        BR2 = flavio.np_prediction('BR(tau->mununu)', w)
        self.assertAlmostEqual(BR1 / BR2, 4, delta=0.2)

    def test_GFeff(self):
        CLSM = -4 *  par['GF'] / sqrt(2)
        w = Wilson({'CVLL_numunueemu': CLSM},
                    80, 'WET', 'flavio')
        wc_np = flavio.WilsonCoefficients.from_wilson(w, par)
        GFeff = taulnunu.GFeff(wc_np, par)
        self.assertEqual(GFeff / par['GF'], 0.5)
        w = Wilson({'CVLL_numunueemu': -0.5 * CLSM},
                    80, 'WET', 'flavio')
        wc_np = flavio.WilsonCoefficients.from_wilson(w, par)
        GFeff = taulnunu.GFeff(wc_np, par)
        self.assertEqual(GFeff / par['GF'], 2)

    def test_taulnunu_np3(self):
        CLSM = -4 *  par['GF'] / sqrt(2)
        w = Wilson({'CVLL_numunueemu': CLSM},
                    80, 'WET', 'flavio')
        BRSM = flavio.sm_prediction('BR(tau->enunu)')
        BRNP = flavio.np_prediction('BR(tau->enunu)', w)
        self.assertEqual(BRNP / BRSM, 0.25)
        w = Wilson({'CVLL_numunueemu': -0.5 * CLSM},
                    80, 'WET', 'flavio')
        BRSM = flavio.sm_prediction('BR(tau->enunu)')
        BRNP = flavio.np_prediction('BR(tau->enunu)', w)
        self.assertEqual(BRNP / BRSM, 4)

    def test_taulnunu_np4(self):
        CLSM = -4 *  par['GF'] / sqrt(2)
        w = Wilson({'CVLL_numunueemu': -0.5 * CLSM,
                    'CVLL_nutaunueetau': CLSM},
                    80, 'WET', 'flavio')
        BR1 = flavio.sm_prediction('BR(tau->enunu)')
        BR2 = flavio.np_prediction('BR(tau->enunu)', w)
        self.assertEqual(BR2 / BR1, 9)
