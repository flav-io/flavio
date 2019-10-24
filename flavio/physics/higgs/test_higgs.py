import unittest
import flavio
import wilson
from math import sqrt


par = flavio.default_parameters.get_central_all()
wc = flavio.WilsonCoefficients()
wc.set_initial({}, 91.1876, 'SMEFT', 'Warsaw')


class TestHiggsProduction(unittest.TestCase):
    def test_mugg(self):
        RSM = flavio.physics.higgs.production.ggF(wc.wc)
        self.assertEqual(RSM, 1)


class TestHiggsDecay(unittest.TestCase):
    def test_mugg(self):
        RSM = flavio.physics.higgs.decay.h_bb(wc.wc)
        self.assertEqual(RSM, 1)


class TestHiggsWidth(unittest.TestCase):
    def test_gamma_h(self):
        RSM = flavio.physics.higgs.width.Gamma_h(par, wc.wc)
        self.assertEqual(RSM, 1)

    def test_gamma_h_quadratic(self):
        # large up quark Yukawa coupling to make h>uu comparable to h>bb
        C = -sqrt(2) * 2.80 / 246.22**3  # -sqrt(2) * m_b(m_h) / v**2
        wc_np = wilson.Wilson({'uphi_11': C}, 125, 'SMEFT',  'Warsaw')
        R = flavio.physics.higgs.width.Gamma_h(par, wc_np.wc)
        self.assertAlmostEqual(R, 1.58, delta=0.15)


class TestHiggsSignalStrengths(unittest.TestCase):
    def test_mugg(self):
        muSM = flavio.physics.higgs.signalstrength.higgs_signalstrength(wc, par, 'ggF', 'h_tautau')
        self.assertEqual(muSM, 1)

    def test_obs_sm_ff(self):
        muSM = flavio.sm_prediction('mu_gg(h->tautau)')
        self.assertEqual(muSM, 1)

    def test_obs_sm_gaga(self):
        muSM = flavio.sm_prediction('mu_gg(h->gammagamma)')
        self.assertEqual(muSM, 1)

    def test_obs_np_ff(self):
        w = wilson.Wilson({'ephi_33': 1e-8}, 125, 'SMEFT', 'Warsaw')
        mu_tau = flavio.np_prediction('mu_gg(h->tautau)', w)
        mu_mu = flavio.np_prediction('mu_gg(h->mumu)', w)
        # mu_mu is not exactly =1 since it modifies the Higgs total width
        self.assertAlmostEqual(mu_mu, 1, delta=0.02)
        self.assertNotAlmostEqual(mu_tau, 1, delta=0.02)

    def test_obs_np_gaga(self):
        w = wilson.Wilson({'phiW': 1e-6}, 125, 'SMEFT', 'Warsaw')
        mu = flavio.np_prediction('mu_gg(h->gammagamma)', w)
        self.assertNotAlmostEqual(mu, 1, delta=0.02)

    def test_noerror(self):
        flavio.sm_prediction('mu_gg(h->Zgamma)')
        flavio.sm_prediction('mu_tth(h->tautau)')
        flavio.sm_prediction('mu_Wh(h->mumu)')
        flavio.sm_prediction('mu_Zh(h->bb)')
        flavio.sm_prediction('mu_VBF(h->WW)')


class TestHiggsMeasurements(unittest.TestCase):
    def test_run1_correlation(self):
        # check that the large anticorrlation between Wh and Zh gaga is correct
        m = flavio.classes.Measurement['LHC Run 1 Higgs combination']
        c = m._constraints[0][0]
        self.assertEqual(len(c.central_value), 20)
        self.assertAlmostEqual(c.correlation[12, 8], -0.64, delta=0.005)
        self.assertEqual(m.all_parameters[8], 'mu_Wh(h->gammagamma)')
        self.assertEqual(m.all_parameters[12], 'mu_Zh(h->gammagamma)')

    def test_atlas_run2_correlation(self):
        # check that the large anticorrlation between ggF and VBF tautau is correct
        m = flavio.classes.Measurement['ATLAS Run 2 Higgs 80/fb']
        c = m._constraints[0][0]
        self.assertEqual(len(c.central_value), 16)
        self.assertAlmostEqual(c.correlation[3, 7], -0.44, delta=0.005)
        self.assertEqual(m.all_parameters[3], 'mu_gg(h->tautau)')
        self.assertEqual(m.all_parameters[7], 'mu_VBF(h->tautau)')


    def test_cms_run2_correlation(self):
        # check that the large anticorrlation between ggF and VBF mumu is correct
        m = flavio.classes.Measurement['CMS Run 2 Higgs 36/fb']
        c = m._constraints[0][0]
        self.assertEqual(len(c.central_value), 24)
        self.assertAlmostEqual(c.correlation[5, 10], -0.54, delta=0.005)
        self.assertEqual(m.all_parameters[5], 'mu_gg(h->mumu)')
        self.assertEqual(m.all_parameters[10], 'mu_VBF(h->mumu)')
