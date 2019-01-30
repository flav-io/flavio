import unittest
import flavio
from .kll import br_kll
from flavio.physics import ckm
from math import pi


constraints = flavio.default_parameters
wc_obj = flavio.WilsonCoefficients()
par = constraints.get_central_all()


class TestKll(unittest.TestCase):
    def test_klmm_sd_sm(self):
        # compare to hep-ph/0605203
        # correct for different CKM choice
        par06 = par.copy()
        par06.update({'Vus': 0.22715, 'Vub': 0.003683, 'Vcb': 0.04161,
                      'delta': 59.0 / 180 * pi})
        self.assertAlmostEqual(br_kll(par06, wc_obj, 'KL', 'mu', 'mu', ld=False),
                               0.79 * 1e-9,
                               delta=0.02e-9)

    def test_ksmm_sd_sm(self):
        # compare to 1707.06999
        # correct for different CKM choice
        my_xi_t = ckm.xi('t', 'sd')(par)
        xi_t = ckm.xi('t', 'sd')({'Vus': 0.22508, 'Vub': 0.003715, 'Vcb': 0.04181,
                                  'delta': 65.4 / 180 * pi})
        r = (my_xi_t.imag / xi_t.imag)**2
        self.assertAlmostEqual(br_kll(par, wc_obj, 'KS', 'mu', 'mu', ld=False),
                               r * 0.19e-12,
                               delta=0.02e-12)

    def test_ksmm_ld_sm(self):
        # 1712.01295 eq. (2.33)
        _par = par.copy()
        _par.update({'Vus': 0.22508, 'Vub': 0.003715, 'Vcb': 0.04181,
                      'delta': 65.4 / 180 * pi})
        self.assertAlmostEqual(br_kll(_par, wc_obj, 'KS', 'mu', 'mu', ld=True)
                               - br_kll(_par, wc_obj, 'KS', 'mu', 'mu', ld=False),
                               4.99e-12,
                               delta=0.1e-12)

    def test_klmm_sm_p(self):
        par_p = par.copy()
        par_p['chi_disp(KL->gammagamma)'] = 0.71
        par_p.update({'Vus': 0.22508, 'Vub': 0.003715, 'Vcb': 0.04181,
                      'delta': 65.4 / 180 * pi})
        self.assertAlmostEqual(br_kll(par_p, wc_obj, 'KL', 'mu', 'mu', ld=True),
                               6.85e-9,
                               delta=0.2e-9)

    def test_klmm_sm_m(self):
        par_m = par.copy()
        par_m['chi_disp(KL->gammagamma)'] = -0.71
        par_m.update({'Vus': 0.22508, 'Vub': 0.003715, 'Vcb': 0.04181,
                      'delta': 65.4 / 180 * pi})
        self.assertAlmostEqual(br_kll(par_m, wc_obj, 'KL', 'mu', 'mu', ld=True),
                               8.11e-9,
                               delta=0.2e-9)

    def test_ksmm_sm(self):
        # 1712.01295 eq. (2.33)
        self.assertAlmostEqual(flavio.sm_prediction('BR(KS->mumu)'),
                               5.2e-12,
                               delta=0.2e-12)
