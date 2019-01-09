import unittest
import flavio


par = flavio.default_parameters.get_central_all()


class TestBetaDecays(unittest.TestCase):
    def test_wceff(self):
        wc_obj = flavio.WilsonCoefficients()
        wceff = flavio.physics.betadecays.common.wc_eff(par, wc_obj, scale=2, nu='e')
        self.assertAlmostEqual(wceff['V'], 2, delta=0.05)
        self.assertAlmostEqual(wceff['A'], -1.27 * 2, delta=0.05)
        self.assertEqual(wceff['S'], 0)
        self.assertEqual(wceff['P'], 0)
        self.assertEqual(wceff['T'], 0)
        wc_obj = flavio.WilsonCoefficients()
        wc_obj.set_initial({'CVL_duenue': 1}, 2, 'WET-3', 'flavio')
        wceff = flavio.physics.betadecays.common.wc_eff(par, wc_obj, scale=2, nu='e')
        self.assertAlmostEqual(wceff['V'], 4, delta=0.05)
        self.assertAlmostEqual(wceff['A'], -1.27 * 4, delta=0.05)
        self.assertEqual(wceff['S'], 0)
        self.assertEqual(wceff['P'], 0)
        self.assertEqual(wceff['T'], 0)
        wc_obj.set_initial({'CVR_duenue': 1}, 2, 'WET-3', 'flavio')
        wceff = flavio.physics.betadecays.common.wc_eff(par, wc_obj, scale=2, nu='e')
        self.assertAlmostEqual(wceff['V'], 4, delta=0.05)
        self.assertAlmostEqual(wceff['A'], 0, delta=0.05)
        self.assertEqual(wceff['S'], 0)
        self.assertEqual(wceff['P'], 0)
        self.assertEqual(wceff['T'], 0)
        wc_obj.set_initial({'CT_duenue': 1}, 2, 'WET-3', 'flavio')
        wceff = flavio.physics.betadecays.common.wc_eff(par, wc_obj, scale=2, nu='e')
        self.assertAlmostEqual(wceff['V'], 2, delta=0.05)
        self.assertAlmostEqual(wceff['A'], -1.27 * 2, delta=0.05)
        self.assertEqual(wceff['S'], 0)
        self.assertEqual(wceff['P'], 0)
        self.assertAlmostEqual(wceff['T'], 2 * 1, delta=0.1)
