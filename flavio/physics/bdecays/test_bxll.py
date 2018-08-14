import unittest
import numpy as np
import flavio
from flavio.physics.bdecays.bxll import _bxll_dbrdq2, bxll_afb_num_int, bxll_afb_den_int

class TestBXll(unittest.TestCase):
    def test_bxll(self):
        # check whether QED corrections have the right behaviour
        wc_obj = flavio.WilsonCoefficients()
        par = flavio.default_parameters.get_central_all()
        br_1_noqedpc =  _bxll_dbrdq2(1, wc_obj, par, 's', 'mu', include_qed=False, include_pc=False)
        br_1_qed =  _bxll_dbrdq2(1, wc_obj, par, 's', 'mu', include_qed=True, include_pc=False)
        br_6_noqedpc =  _bxll_dbrdq2(6, wc_obj, par, 's', 'mu', include_qed=False, include_pc=False)
        br_6_qed =  _bxll_dbrdq2(6, wc_obj, par, 's', 'mu', include_qed=True, include_pc=False)
        br_15_noqedpc =  _bxll_dbrdq2(15, wc_obj, par, 's', 'mu', include_qed=False, include_pc=False)
        br_15_qed =  _bxll_dbrdq2(15, wc_obj, par, 's', 'mu', include_qed=True, include_pc=False)
        br_21_noqedpc =  _bxll_dbrdq2(21, wc_obj, par, 's', 'mu', include_qed=False, include_pc=False)
        br_21_qed =  _bxll_dbrdq2(21, wc_obj, par, 's', 'mu', include_qed=True, include_pc=False)
        self.assertAlmostEqual((br_1_qed+br_6_qed)/(br_1_noqedpc+br_6_noqedpc),
                                1.02, delta=0.01) # should lead to a 2% enhancement
        self.assertAlmostEqual((br_15_qed+br_21_qed)/(br_15_noqedpc+br_21_noqedpc),
                                0.92, delta=0.03) # should lead to a 8% suppression


        # compare SM predictions to arXiv:1503.04849
        # to convert to the parameters used there
        xi_t = flavio.physics.ckm.xi('t','bs')(par)
        Vcb = flavio.physics.ckm.get_ckm(par)[1,2]
        r = abs(xi_t)**2/Vcb**2/0.9621*0.574/par['C_BXlnu']*par['BR(B->Xcenu)_exp']/0.1051
        self.assertAlmostEqual(1e6*flavio.sm_prediction('<BR>(B->Xsmumu)', 1, 3.5)/r,
                               0.888, delta=0.02)
        self.assertAlmostEqual(1e6*flavio.sm_prediction('<BR>(B->Xsmumu)', 3.5, 6)/r,
                               0.731, delta=0.01)
        self.assertAlmostEqual(1e7*flavio.sm_prediction('<BR>(B->Xsmumu)', 14.4, 25)/r,
                               2.53, delta=0.6) # larger difference due to Krüger-Sehgal
        self.assertAlmostEqual(1e6*flavio.sm_prediction('<BR>(B->Xsee)', 1, 3.5)/r,
                               0.926, delta=0.04)
        self.assertAlmostEqual(1e6*flavio.sm_prediction('<BR>(B->Xsee)', 3.5, 6)/r,
                               0.744, delta=0.015)
        self.assertAlmostEqual(1e7*flavio.sm_prediction('<BR>(B->Xsee)', 14.4, 25)/r,
                               2.20, delta=0.6) # larger difference due to Krüger-Sehgal

    def test_bxll_lratio(self):
        # compare to arXiv:1503.04849
        self.assertAlmostEqual(flavio.sm_prediction('<Rmue>(B->Xsll)', 1, 3.5),
                               0.96, delta=0.01)
        self.assertAlmostEqual(flavio.sm_prediction('<Rmue>(B->Xsll)', 14.4, 25),
                               1.15, delta=0.01)
        # for tau, just check this doesn't raise an error
        flavio.sm_prediction('<Rtaumu>(B->Xsll)', 14.4, 25)

    def test_bxll_afb(self):
        # check calling outside of kinematical regions yields 0
        self.assertAlmostEqual(flavio.sm_prediction('AFB(B->Xsmumu)', 0), 0)
        self.assertAlmostEqual(flavio.sm_prediction('AFB(B->Xsll)', 30), 0)
        # just check differential AFB doesn't raise errors
        flavio.sm_prediction('AFB(B->Xsee)', 1)
        flavio.sm_prediction('AFB(B->Xsmumu)', 6)
        flavio.sm_prediction('AFB(B->Xsll)', 14.4)
        # check whether QED corrections have the right behaviour
        # (table 2 of arXiv:1503.04849)
        wc_obj = flavio.WilsonCoefficients()
        par = flavio.default_parameters.get_central_all()
        afb_num_low1_noqed = bxll_afb_num_int(1, 3.5, wc_obj, par, 's', 'e', include_qed=False)
        afb_num_low1_qed   = bxll_afb_num_int(1, 3.5, wc_obj, par, 's', 'e', include_qed=True)
        afb_num_low2_noqed = bxll_afb_num_int(3.5, 6, wc_obj, par, 's', 'e', include_qed=False)
        afb_num_low2_qed   = bxll_afb_num_int(3.5, 6, wc_obj, par, 's', 'e', include_qed=True)
        afb_den_low1_noqed = bxll_afb_den_int(1, 3.5, wc_obj, par, 's', 'e', include_qed=False)
        afb_den_low1_qed   = bxll_afb_den_int(1, 3.5, wc_obj, par, 's', 'e', include_qed=True)
        afb_den_low2_noqed = bxll_afb_den_int(3.5, 6, wc_obj, par, 's', 'e', include_qed=False)
        afb_den_low2_qed   = bxll_afb_den_int(3.5, 6, wc_obj, par, 's', 'e', include_qed=True)
        self.assertAlmostEqual((afb_num_low1_qed-afb_num_low1_noqed)/afb_num_low1_qed,
                                -0.107, delta=0.050) # should lead to a -10.7% suppression
        self.assertAlmostEqual((afb_num_low2_qed-afb_num_low2_noqed)/afb_num_low2_qed,
                                +0.162, delta=0.020) # should lead to a 16.2% enhancement
        self.assertAlmostEqual((afb_den_low1_qed-afb_den_low1_noqed)/afb_den_low1_qed,
                                0.068, delta=0.005) # should lead to a 6.8% enhancement
        self.assertAlmostEqual((afb_den_low2_qed-afb_den_low2_noqed)/afb_den_low2_qed,
                                0.031, delta=0.010) # should lead to a 3.1% enhancement
        # compare SM predictions to arXiv:1503.04849
        self.assertAlmostEqual(
            flavio.sm_prediction('<AFB>(B->Xsee)', 1, 3.5)/(3/4.*(-1.03e-7)/((2.91e-7)+(6.35e-7))),
            1, delta=0.15)
        self.assertAlmostEqual(
            flavio.sm_prediction('<AFB>(B->Xsee)', 3.5, 6)/(3/4.*(0.73e-7)/((2.43e-7)+(4.97e-7))),
            1, delta=0.1)
        self.assertAlmostEqual(
            flavio.sm_prediction('<AFB>(B->Xsmumu)', 1, 3.5)/(3/4.*(-1.10e-7)/((2.09e-7)+(6.79e-7))),
            1, delta=0.1)
        self.assertAlmostEqual(
            flavio.sm_prediction('<AFB>(B->Xsmumu)', 3.5, 6)/(3/4.*(0.67e-7)/((1.94e-7)+(5.34e-7))),
            1, delta=0.1)
