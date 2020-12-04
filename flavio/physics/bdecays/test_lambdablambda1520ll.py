import unittest
import numpy as np
import flavio
import ckmutil.ckm

wc_sm = flavio.WilsonCoefficients()
par_nominal = flavio.default_parameters.copy()
par_nominal.set_constraint('alpha_e', 1/128.940)
par = par_nominal.get_central_all()


wc_np_mu = flavio.WilsonCoefficients()
wc_np_mu.set_initial({'C9_bsmumu': -1.11}, scale=160)
wc_np_mu.get_wc(sector='bsmumu', scale=4.8, par=par, nf_out=5)


def ass_sm(s, wc, name, q2min, q2max, target, delta, scalef=1):
    obs = flavio.classes.Observable[name]
    c = obs.prediction_central(par_nominal, wc, q2min, q2max)*scalef
    s.assertAlmostEqual(c, target, delta=delta)

    
class TestLambdabLambdall(unittest.TestCase):
    def test_params(self):
        self.assertAlmostEqual(par['m_W'], 80.399, delta=0.023)
        self.assertEqual(par['m_Z'], 91.1876)
        self.assertAlmostEqual(par['s2w'], 0.2313, delta=0.000011)
        self.assertAlmostEqual(par['alpha_s'], 0.1184, delta=0.0007)
        self.assertEqual(par['alpha_e'], 1/128.940)
        laC, A, rhobar, etabar = flavio.physics.ckm.tree_to_wolfenstein(par['Vus'], par['Vub'], par['Vcb'], par['delta'])
        self.assertAlmostEqual(laC, 0.22546, delta=0.0008) 
        #self.assertAlmostEqual(A, 0.805, delta=0.020) 
        self.assertAlmostEqual(rhobar, 0.144, delta=0.025) 
        self.assertAlmostEqual(etabar, 0.342, delta=0.016) 
        
        
    def test_lambdablambdall_SM(self):
        # compare to SM values assuming 10% uncertainty on form factors in table 1 of 2005.09602
        #ass_sm(self, wc_sm, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 0.1, 3, 0.397, 0.054, 1e9)
        ass_sm(self, wc_sm, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 3, 6, 1.29, 0.18, 1e9)
        ass_sm(self, wc_sm, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 6, 8.68, 3.22, 0.42, 1e9)
        ass_sm(self, wc_sm, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.95, 0.13, 1e9)
        #ass_sm(self, wc_sm, '<AFBl>(Lambdab->Lambda(1520)mumu)', 0.1, 3, 0.048, 0.018)
        ass_sm(self, wc_sm, '<AFBl>(Lambdab->Lambda(1520)mumu)', 3, 6, -0.127, 0.033)
        ass_sm(self, wc_sm, '<AFBl>(Lambdab->Lambda(1520)mumu)', 6, 8.68, -0.235, 0.040)
        ass_sm(self, wc_sm, '<AFBl>(Lambdab->Lambda(1520)mumu)', 1, 6, -0.098, 0.031)
        #ass_sm(self, wc_sm, '<S_1cc>(Lambdab->Lambda(1520)mumu)', 0.1, 3, 0.181, 0.031)
        #ass_sm(self, wc_sm, '<S_1cc>(Lambdab->Lambda(1520)mumu)', 3, 6, 0.242, 0.042)
        #ass_sm(self, wc_sm, '<S_1cc>(Lambdab->Lambda(1520)mumu)', 6, 8.68, 0.361, 0.051)
        #ass_sm(self, wc_sm, '<S_1cc>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.221, 0.038)

        
    def test_lambdablambdall_NP(self):
        # compare to NP values assuming 10% uncertainty on form factors in table 1 of 2005.09602
        #ass_sm(self, wc_np_mu, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 0.1, 3, 0.337, 0.042, 1e9)
        ass_sm(self, wc_np_mu, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 3, 6, 1.04, 0.13, 1e9)
        ass_sm(self, wc_np_mu, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 6, 8.68, 2.58, 0.32, 1e9)
        ass_sm(self, wc_np_mu, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.77, 0.10, 1e9)
        #ass_sm(self, wc_np_mu, '<AFBl>(Lambdab->Lambda(1520)mumu)', 0.1, 3, 0.098, 0.022)
        ass_sm(self, wc_np_mu, '<AFBl>(Lambdab->Lambda(1520)mumu)', 3, 6, -0.059, 0.034)
        ass_sm(self, wc_np_mu, '<AFBl>(Lambdab->Lambda(1520)mumu)', 6, 8.68, -0.166, 0.041)
        ass_sm(self, wc_np_mu, '<AFBl>(Lambdab->Lambda(1520)mumu)', 1, 6, -0.031, 0.032)
        #ass_sm(self, wc_np_mu, '<S_1cc>(Lambdab->Lambda(1520)mumu)', 0.1, 3, 0.240, 0.038)
        #ass_sm(self, wc_np_mu, '<S_1cc>(Lambdab->Lambda(1520)mumu)', 3, 6, 0.263, 0.042)
        #ass_sm(self, wc_np_mu, '<S_1cc>(Lambdab->Lambda(1520)mumu)', 6, 8.68, 0.371, 0.050)
        #ass_sm(self, wc_np_mu, '<S_1cc>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.246, 0.039)

