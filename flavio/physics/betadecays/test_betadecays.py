import unittest
import flavio
from flavio.physics.units import s


par = flavio.default_parameters.get_central_all()


class TestBetaDecays(unittest.TestCase):
    def test_wceff(self):
        wc_obj = flavio.WilsonCoefficients()
        wceff = flavio.physics.betadecays.common.wc_eff(par, wc_obj, scale=1, nu='e')
        self.assertAlmostEqual(wceff['V'], 1, delta=0.05)
        self.assertAlmostEqual(wceff['A'], -1.27, delta=0.05)
        self.assertEqual(wceff['S'], 0)
        self.assertEqual(wceff['P'], 0)
        self.assertEqual(wceff['T'], 0)
        wc_obj = flavio.WilsonCoefficients()
        wc_obj.set_initial({'CVL_duenue': 1}, 1, 'WET-3', 'flavio')
        wceff = flavio.physics.betadecays.common.wc_eff(par, wc_obj, scale=1, nu='e')
        self.assertAlmostEqual(wceff['V'], 2, delta=0.05)
        self.assertAlmostEqual(wceff['A'], -1.27 * 2, delta=0.05)
        self.assertEqual(wceff['S'], 0)
        self.assertEqual(wceff['P'], 0)
        self.assertEqual(wceff['T'], 0)
        wc_obj.set_initial({'CVR_duenue': 1}, 1, 'WET-3', 'flavio')
        wceff = flavio.physics.betadecays.common.wc_eff(par, wc_obj, scale=1, nu='e')
        self.assertAlmostEqual(wceff['V'], 2, delta=0.05)
        self.assertAlmostEqual(wceff['A'], 0, delta=0.05)
        self.assertEqual(wceff['S'], 0)
        self.assertEqual(wceff['P'], 0)
        self.assertEqual(wceff['T'], 0)
        wc_obj.set_initial({'CT_duenue': 1}, 1, 'WET-3', 'flavio')
        wceff = flavio.physics.betadecays.common.wc_eff(par, wc_obj, scale=1, nu='e')
        self.assertAlmostEqual(wceff['V'], 1, delta=0.05)
        self.assertAlmostEqual(wceff['A'], -1.27, delta=0.05)
        self.assertEqual(wceff['S'], 0)
        self.assertEqual(wceff['P'], 0)
        self.assertAlmostEqual(wceff['T'], 4 * 1, delta=0.1)

    def test_ft(self):
        # compare to exp values in table 4 of 1803.08732
        wc_obj = flavio.WilsonCoefficients()
        Ft = flavio.physics.betadecays.ft.Ft(par, wc_obj, '10C')
        self.assertAlmostEqual(Ft / s, 3078, delta=2 * 5)
        Ft = flavio.physics.betadecays.ft.Ft(par, wc_obj, '26mAl')
        self.assertAlmostEqual(Ft / s, 3072.9, delta=3 * 1)
        Ft = flavio.physics.betadecays.ft.Ft(par, wc_obj, '46V')
        self.assertAlmostEqual(Ft / s, 3074.1, delta=2 * 2)
        Ft = flavio.sm_prediction('Ft(38Ca)')
        self.assertAlmostEqual(Ft / s, 3076.4, delta=2 * 7.2)
