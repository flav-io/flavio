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
        Ft = flavio.physics.betadecays.ft.Ft_superallowed(par, wc_obj, '10C')
        self.assertAlmostEqual(Ft / s, 3078, delta=2 * 5)
        Ft = flavio.physics.betadecays.ft.Ft_superallowed(par, wc_obj, '26mAl')
        self.assertAlmostEqual(Ft / s, 3072.9, delta=3 * 1)
        Ft = flavio.physics.betadecays.ft.Ft_superallowed(par, wc_obj, '46V')
        self.assertAlmostEqual(Ft / s, 3074.1, delta=2 * 2)
        Ft = flavio.sm_prediction('Ft(38Ca)')
        self.assertAlmostEqual(Ft / s, 3076.4, delta=2 * 7.2)

    def test_taun(self):
        # compare to exp value in table 5 of 1803.08732
        tau_n = flavio.sm_prediction('tau_n', me_E=0.655)
        self.assertAlmostEqual(tau_n / s, 879.75, delta=39)

    def test_corrn(self):
        # compare to exp values in table 5 of 1803.08732
        self.assertAlmostEqual(flavio.sm_prediction('a_n'), -0.1034, delta=2 * 0.0037)
        self.assertAlmostEqual(flavio.sm_prediction('atilde_n', me_E=0.695), -0.1090, delta=0.01)
        self.assertAlmostEqual(flavio.sm_prediction('Atilde_n', me_E=0.569), -0.11869, delta=0.01)
        self.assertAlmostEqual(flavio.sm_prediction('Btilde_n', me_E=0.591), 0.9805, delta=3 * 0.003)
        self.assertAlmostEqual(flavio.sm_prediction('lambdaAB_n', me_E=0.581), -1.2686, delta=0.04)
        self.assertEqual(flavio.sm_prediction('D_n'), 0)
        self.assertEqual(flavio.sm_prediction('R_n'), 0)
