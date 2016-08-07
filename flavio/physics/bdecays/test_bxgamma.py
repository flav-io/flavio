import unittest
import numpy as np
import flavio

wc_obj = flavio.WilsonCoefficients()
par = flavio.default_parameters
par_dict = par.get_central_all()

class TestBXgamma(unittest.TestCase):
    def test_bxgamma(self):
        # compare SM predictions to arXiv:1503.01789
        self.assertAlmostEqual(1e4*flavio.sm_prediction('BR(B->Xsgamma)'),
                               3.36,
                               delta=0.2)
        self.assertAlmostEqual(1e5*flavio.sm_prediction('BR(B->Xdgamma)'),
                               1.73,
                               delta=0.2)

    def test_acp(self):
        # check that the SM central values for the individual B->Xsgamma
        # and B->Xdgamma (ignoring long distance contributions) roughly
        # agree with the values quoted in hep-ph/0312260
        wc_sm_s = flavio.physics.bdecays.wilsoncoefficients.wctot_dict(wc_obj, 'bsee', scale=2, par=par_dict, nf_out=5)
        wc_sm_d = flavio.physics.bdecays.wilsoncoefficients.wctot_dict(wc_obj, 'bdee', scale=2, par=par_dict, nf_out=5)
        p_ave_s = flavio.physics.bdecays.bxgamma.PE0_BR_BXgamma(wc_sm_s, par_dict, 's', 1.6)
        p_asy_s = flavio.physics.bdecays.bxgamma.PE0_ACP_BXgamma(wc_sm_s, par_dict, 's', 1.6)
        acp_s = p_asy_s/p_ave_s
        self.assertAlmostEqual(100*acp_s, 0.44, delta=0.5)
        p_ave_d = flavio.physics.bdecays.bxgamma.PE0_BR_BXgamma(wc_sm_d, par_dict, 'd', 1.6)
        p_asy_d = flavio.physics.bdecays.bxgamma.PE0_ACP_BXgamma(wc_sm_d, par_dict, 'd', 1.6)
        acp_d = p_asy_d/p_ave_d
        # check that the s+d CP asymmetry vanishes
        self.assertAlmostEqual(flavio.sm_prediction('ACP(B->Xgamma)'), 0, delta=1e-9)
