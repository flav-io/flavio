import unittest
import numpy as np
import flavio
from flavio.physics.bdecays.bxll import _bxll_dbrdq2

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
                               0.888, delta=0.03)
        self.assertAlmostEqual(1e6*flavio.sm_prediction('<BR>(B->Xsmumu)', 3.5, 6)/r,
                               0.731, delta=0.01)
        self.assertAlmostEqual(1e7*flavio.sm_prediction('<BR>(B->Xsmumu)', 14.4, 25)/r,
                               2.53, delta=0.6) # larger difference due to Krüger-Sehgal
        self.assertAlmostEqual(1e6*flavio.sm_prediction('<BR>(B->Xsee)', 1, 3.5)/r,
                               0.926, delta=0.04)
        self.assertAlmostEqual(1e6*flavio.sm_prediction('<BR>(B->Xsee)', 3.5, 6)/r,
                               0.744, delta=0.01)
        self.assertAlmostEqual(1e7*flavio.sm_prediction('<BR>(B->Xsee)', 14.4, 25)/r,
                               2.20, delta=0.6) # larger difference due to Krüger-Sehgal

    def test_bxll_lratio(self):
        # compare to arXiv:1503.04849
        self.assertAlmostEqual(flavio.sm_prediction('<Rmue>(B->Xsll)', 1, 3.5),
                               0.96, delta=0.02)
        self.assertAlmostEqual(flavio.sm_prediction('<Rmue>(B->Xsll)', 14.4, 25),
                               1.15, delta=0.02)
        # for tau, just check this doesn't raise an error
        flavio.sm_prediction('<Rtaumu>(B->Xsll)', 14.4, 25)
