import unittest
import flavio
from flavio.physics.zdecays.gammazsm import Zobs, pb


par = flavio.default_parameters.get_central_all()


class TestGammaZ(unittest.TestCase):
    def test_obs_sm(self):
        # check the SM predictions
        self.assertAlmostEqual(flavio.sm_prediction('GammaZ'),
                               2.4950, delta=0.001)
        self.assertAlmostEqual(flavio.sm_prediction('GammaZ'),
                               1 / par['tau_Z'], delta=0.001)
        self.assertAlmostEqual(flavio.sm_prediction('sigma_had') / pb / 1e3,
                               41.488, delta=0.05)
        self.assertAlmostEqual(flavio.sm_prediction('Gamma(Z->ee)'),
                               83.966e-3, delta=0.001e-3)
        self.assertAlmostEqual(flavio.sm_prediction('Gamma(Z->mumu)'),
                               83.966e-3, delta=0.001e-3)
        self.assertAlmostEqual(flavio.sm_prediction('Gamma(Z->tautau)'),
                               83.776e-3, delta=0.001e-3)
        self.assertAlmostEqual(flavio.sm_prediction('Gamma(Z->uu)'),
                               299.936e-3, delta=0.01e-3)
        self.assertAlmostEqual(flavio.sm_prediction('Gamma(Z->cc)'),
                               299.860e-3, delta=0.01e-3)
        self.assertAlmostEqual(flavio.sm_prediction('Gamma(Z->dd)'),
                               382.770e-3, delta=0.01e-3)
        self.assertAlmostEqual(flavio.sm_prediction('Gamma(Z->ss)'),
                               382.770e-3, delta=0.01e-3)
        self.assertAlmostEqual(flavio.sm_prediction('Gamma(Z->bb)'),
                               375.724e-3, delta=0.02e-3)
        self.assertAlmostEqual(flavio.sm_prediction('Gamma(Z->nunu)'),
                               167.157e-3, delta=0.01e-3)
        self.assertAlmostEqual(flavio.sm_prediction('R_l'),
                               20750.9e-3, delta=1e-3)
        self.assertAlmostEqual(flavio.sm_prediction('R_c'),
                               172.23e-3, delta=0.01e-3)
        self.assertAlmostEqual(flavio.sm_prediction('R_b'),
                               215.80e-3, delta=0.01e-3)
        self.assertAlmostEqual(flavio.sm_prediction('R_e'),
                               20.743, delta=0.01)
        self.assertAlmostEqual(flavio.sm_prediction('R_mu'),
                               20.743, delta=0.01)
        self.assertAlmostEqual(flavio.sm_prediction('R_tau'),
                               20.743, delta=0.05)

    def test_r_sm(self):
        # check that the Sm predictions for the Ri agree with the Gammas
        par = flavio.default_parameters.get_central_all()
        mh = par['m_h']
        mt = par['m_t']
        als = par['alpha_s']
        Da = 0.059
        mZ = par['m_Z']
        arg = (mh, mt, als, Da, mZ)
        Rl = Zobs('Rl', *arg)
        Rc = Zobs('Rc', *arg)
        Rb = Zobs('Rb', *arg)
        Ge = Zobs('Gammae,mu', *arg)
        Gmu = Zobs('Gammae,mu', *arg)
        Gtau = Zobs('Gammatau', *arg)
        Gu = Zobs('Gammau', *arg)
        Gd = Zobs('Gammad,s', *arg)
        Gs = Zobs('Gammad,s', *arg)
        Gc = Zobs('Gammac', *arg)
        Gb = Zobs('Gammab', *arg)
        Ghad = Gu + Gd + Gc + Gs + Gb
        Gl = (Ge + Gmu + Gtau) / 3.
        self.assertAlmostEqual(Rl, Ghad / Gl, delta=1e-4)
        self.assertAlmostEqual(Rc, Gc / Ghad, delta=1e-4)
        self.assertAlmostEqual(Rb, Gb / Ghad, delta=1e-4)

    def test_obs_sm_fv(self):
        # check the SM predictions for FV decays
        sec = [['u', 'c'], ['d', 's', 'b'], ['e', 'mu', 'tau']]
        for f in sec:
            for f1 in f:
                for f2 in f:
                    if f1 != f2:
                        self.assertEqual(flavio.sm_prediction('BR(Z->{}{})'.format(f1, f2)),
                                         0,
                                         msg="Failed BR(Z->{}{})")

class TestAFBZ(unittest.TestCase):
    def test_afbz_sm(self):
        for l in ['e', 'mu', 'tau']:
            self.assertAlmostEqual(flavio.sm_prediction('A(Z->{}{})'.format(l, l)),
                                   0.1472, delta=0.0002, msg="Failed for {}".format(l))
            self.assertAlmostEqual(flavio.sm_prediction('AFB(Z->{}{})'.format(l, l)),
                                   0.0163, delta=0.0002, msg="Failed for {}".format(l))
        self.assertAlmostEqual(flavio.sm_prediction('A(Z->bb)'),
                               0.935, delta=0.001)
        self.assertAlmostEqual(flavio.sm_prediction('A(Z->cc)'),
                               0.668, delta=0.001)
        self.assertAlmostEqual(flavio.sm_prediction('A(Z->ss)'),
                               0.935, delta=0.001)
        self.assertAlmostEqual(flavio.sm_prediction('AFB(Z->bb)'),
                               0.1032, delta=0.0002)
        self.assertAlmostEqual(flavio.sm_prediction('AFB(Z->cc)'),
                               0.0738, delta=0.0002)
