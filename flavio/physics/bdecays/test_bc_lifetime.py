import unittest
import flavio


par = flavio.default_parameters.get_central_all()


class TestTauBc(unittest.TestCase):
    def test_sm(self):
        self.assertEqual(flavio.sm_prediction('tau_Bc'), par['tau_Bc_SM'])
        self.assertAlmostEqual(flavio.sm_prediction('tau_Bc'), par['tau_Bc'],
                               delta=0.1 * par['tau_Bc'])

    def test_np(self):
        wc = flavio.WilsonCoefficients()
        for l in ['e', 'mu', 'tau']:
            wc.set_initial({'CSL_bc' + l + 'nu' + l: 1}, 4.8)
            self.assertTrue(par['tau_Bc_SM'] / flavio.np_prediction('tau_Bc', wc) > 1.1)
            self.assertAlmostEqual(par['tau_Bc_SM'] / flavio.np_prediction('tau_Bc', wc),
                                   1 + flavio.np_prediction('BR(Bc->' + l + 'nu)', wc)
                                   - flavio.sm_prediction('BR(Bc->' + l + 'nu)'),
                                   delta=0.05,
                                   msg="Failed for {}".format(l))

    def test_exp(self):
        self.assertAlmostEqual(flavio.combine_measurements('tau_Bc').central_value
                               / par['tau_Bc'], 1,
                               delta=0.001)
