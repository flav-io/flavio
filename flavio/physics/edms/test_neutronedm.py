import unittest
import flavio
from wilson import wcxf
from math import sqrt


par = flavio.default_parameters.get_central_all()


class TestNeutronEDM(unittest.TestCase):
    def test_nedm_sm(self):
        self.assertEqual(flavio.sm_prediction('d_n'), 0)

    def test_nedm_jms_G(self):
        wc = wcxf.WC('WET', 'JMS', 160, {'Gtilde': 1e-6})
        wcf = flavio.WilsonCoefficients()
        wcf.set_initial_wcxf(wc)
        wcd = wcf.get_wc('dF=0', scale=2, par=par, eft='WET-3', basis='flavio')
        self.assertEqual(wcd['CG'], 0)
        self.assertAlmostEqual(wcd['CGtilde'], 0.007, delta=0.001)
        par_G = par.copy()
        par_G['gT_d'] = 0
        par_G['nEDM ~rho_d'] = 0
        par_G['gT_u'] = 0
        par_G['nEDM ~rho_u'] = 0
        p = flavio.Observable['d_n'].prediction_par(par_G, wcf)
        self.assertAlmostEqual(p, 4.5e-9, delta=1e-9)

    def test_nedm_jms_CEDM(self):
        wc = wcxf.WC('WET', 'JMS', 160, {'dG_11': {'Im': 1e-10}})
        wcf = flavio.WilsonCoefficients()
        wcf.set_initial_wcxf(wc)
        wcd = wcf.get_wc('dF=0', scale=2, par=par, eft='WET-3', basis='flavio')
        self.assertAlmostEqual(wcd['C8_dd'] / 1j, 0.07, delta=0.02)
        par_dG = par.copy()
        par_dG['gT_d'] = 0
        par_dG['nEDM beta_G'] = 0
        par_dG['gT_u'] = 0
        par_dG['nEDM ~rho_u'] = 0
        p = flavio.Observable['d_n'].prediction_par(par_dG, wcf)
        v = 246.22
        self.assertAlmostEqual(p,
                               abs(-2*par_dG['nEDM ~rho_d']*1e-10),
                               delta=0.3e-10)
