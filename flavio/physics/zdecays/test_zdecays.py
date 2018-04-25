import unittest
import flavio
from math import sqrt, pi
from flavio.physics.zdecays.gammazsm import Zobs, pb
from flavio.physics.zdecays.gammaz import GammaZ_NP
from flavio.physics.zdecays import smeftew


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
        # check the SM predictions for LFV decays
        self.assertEqual(flavio.sm_prediction('BR(Z->emu)'), 0)
        self.assertEqual(flavio.sm_prediction('BR(Z->etau)'), 0)
        self.assertEqual(flavio.sm_prediction('BR(Z->mutau)'), 0)

    def test_Gamma_NP(self):
        # compare NP contributions to A.49-A.52 from 1706.08945
        GF, mZ, s2w_eff = par['GF'], par['m_Z'], par['s2w']*1.0010
        d_gV = 0.055
        d_gA = 0.066
        # A.49-A.52 from 1706.08945
        dGamma_Zll = sqrt(2)*GF*mZ**3/(6*pi) * (-d_gA + (-1+4*s2w_eff)*d_gV)
        dGamma_Znn = sqrt(2)*GF*mZ**3/(6*pi) * (d_gA + d_gV)
        dGamma_Zuu = sqrt(2)*GF*mZ**3/(pi) * (d_gA -1/3*(-3+8*s2w_eff)*d_gV) /2
        dGamma_Zdd = sqrt(2)*GF*mZ**3/(pi) * (-3/2*d_gA +1/2*(-3+4*s2w_eff)*d_gV) /3
        # term squared in d_gV and d_gA not included in 1706.08945
        d_g_squared = sqrt(2)*GF*mZ**3/(3*pi)*(abs(d_gV)**2+abs(d_gA)**2)
        self.assertAlmostEqual(
            dGamma_Zll + d_g_squared,
            GammaZ_NP(par, 1, smeftew.gV_SM('e', par), d_gV,
                              smeftew.gA_SM('e', par), d_gA)
        )
        self.assertAlmostEqual(
            dGamma_Znn + d_g_squared,
            GammaZ_NP(par, 1, smeftew.gV_SM('nue', par), d_gV,
                              smeftew.gA_SM('nue', par), d_gA)
        )
        self.assertAlmostEqual(
            dGamma_Zuu + 3*d_g_squared,
            GammaZ_NP(par, 3, smeftew.gV_SM('u', par), d_gV,
                              smeftew.gA_SM('u', par), d_gA)
        )
        self.assertAlmostEqual(
            dGamma_Zdd + 3*d_g_squared,
            GammaZ_NP(par, 3, smeftew.gV_SM('d', par), d_gV,
                              smeftew.gA_SM('d', par), d_gA)
        )

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
