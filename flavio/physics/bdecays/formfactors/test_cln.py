"""Compare to parameters of https://arxiv.org/abs/1908.09398

Cf.
https://github.com/eos/eos/blob/8487191336d608fb9517979d84188420834d568a/eos/form-factors/mesonic-hqet_TEST.cc#L199-L358
https://github.com/eos/eos/blob/8487191336d608fb9517979d84188420834d568a/eos/form-factors/mesonic-hqet_TEST.cc#L1023-L1180
"""

import unittest
import flavio


par = flavio.default_parameters.copy()
par.set_constraint("CLN rho2_xi", 1.5)
par.set_constraint("CLN c_xi", +3.0 / 2)
par.set_constraint("CLN xi3", +6.0)
#par.set_constraint("xipppp(1)", -9.0)
par.set_constraint("chi_2(1)", +0.5)
par.set_constraint("chi_2p(1)", -1.0)
par.set_constraint("chi_2pp(1)", +2.0)
par.set_constraint("chi_3p(1)", -1.5)
par.set_constraint("chi_3pp(1)", +2.5)
par.set_constraint("eta(1)", +0.25)
par.set_constraint("etap(1)", -1.25)
par.set_constraint("etapp(1)", +1.75)
par.set_constraint("CLN l_1(1)", +0.5)
par.set_constraint("CLN l_2(1)", -2.0)
par.set_constraint("CLN l_3(1)", 0.0)
par.set_constraint("CLN l_4(1)", 0.0)
par.set_constraint("CLN l_5(1)", 0.0)
par.set_constraint("CLN l_6(1)", 0.0)
par.set_constraint("CLN lp_1(1)", 0.0)
par.set_constraint("CLN lp_2(1)", 0.0)
par.set_constraint("CLN lp_3(1)", 0.0)
par.set_constraint("CLN lp_4(1)", 0.0)
par.set_constraint("CLN lp_5(1)", 0.0)
par.set_constraint("CLN lp_6(1)", 0.0)
par.set_constraint("m_B0", 5.27942)
par.set_constraint("m_B+", 5.27942)
par.set_constraint("m_D+", 1.86723)
par.set_constraint("m_D0", 1.86723)
par.set_constraint("lambda_1", -0.30)
par.set_constraint("m_b", 4.2226)  # to reproduce mb1S=4.71 GeV

class TestCLNParam(unittest.TestCase):
    def test_btop(self):
        eps = 0.002
        def ff(q2):
            _par = par.get_central_all()
            return flavio.physics.bdecays.formfactors.b_p.cln.ff('B->D',
                q2, _par, scale=4.8,
                order_z=3, order_z_slp=1, order_z_sslp=1)

        self.assertAlmostEqual(ff(4)['f+'], -0.317099, delta=eps);
        self.assertAlmostEqual(ff(4)['f0'], -0.311925, delta=eps);
        self.assertAlmostEqual(ff(4)['fT'], -0.043808, delta=eps);

        self.assertAlmostEqual(ff(8)['f+'], +0.273187, delta=eps);
        self.assertAlmostEqual(ff(8)['f0'], +0.198352, delta=eps);
        self.assertAlmostEqual(ff(8)['fT'], +0.514150, delta=eps);

        self.assertAlmostEqual(ff(10)['f+'], +0.721643, delta=eps);
        self.assertAlmostEqual(ff(10)['f0'], +0.544512, delta=eps);
        self.assertAlmostEqual(ff(10)['fT'], +0.952830, delta=eps);


    def test_btov(self):
        eps=0.002

        def ff(q2):
            _par = par.get_central_all()
            return flavio.physics.bdecays.formfactors.b_v.cln.ff('B->D*', q2, _par, scale=4.8,
                order_z=3, order_z_slp=1, order_z_sslp=1)

        mB = par.get_central('m_B0')
        mV = par.get_central('m_D*+')

        def q2_w(w):
            return mB**2 + mV**2 - w * (2 * mB * mV)

        q2 = q2_w(w=1.0)
        h = {
            'A1': +0.899905,
            'A2': +0.036348,
            'A3': +0.552732,
            'V':  +1.217624,
            'T1': +0.961994,
            'T2': -0.198494,
            'T3': -0.665817
        }
        ff_eos = flavio.physics.bdecays.formfactors.b_v.cln.h_to_A(mB, mV, h, q2)
        ff_flav = ff(q2)
        for k in ff_flav:
            self.assertAlmostEqual(ff_eos[k], ff_flav[k], delta=eps, msg="Failed for w=1.0, {}".format(k))

        q2 = q2_w(w=1.4)
        h = {
            'A1': +0.684812,
            'A2': -0.075634,
            'A3': +0.702037,
            'V':  +0.896998,
            'T1': +0.686863,
            'T2': -0.104330,
            'T3': -0.195575
        }
        ff_eos = flavio.physics.bdecays.formfactors.b_v.cln.h_to_A(mB, mV, h, q2)
        ff_flav = ff(q2)
        for k in ff_flav:
            self.assertAlmostEqual(ff_eos[k], ff_flav[k], delta=eps, msg="Failed for w=1.4, {}".format(k))