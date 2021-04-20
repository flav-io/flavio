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

    
class TestLambdabLambda1520ll(unittest.TestCase):
    def test_lambdablambda1520ll_SM(self):
        # compare to SM values assuming 10% uncertainty on form factors in table 1 of 2005.09602
        # Differences due to C9eff = C9 and C7eff = C7
        # Second test of angular distribution using lattice QCD form factors
        # in /bdecays/formfactors/lambdab_32/test_lambdab.py 
        ass_sm(self, wc_sm, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 0.1, 3, 0.42, 0.05, 1e9)
        ass_sm(self, wc_sm, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 3, 6, 1.34, 0.16, 1e9)
        ass_sm(self, wc_sm, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 6, 8.68, 3.4, 0.4, 1e9)
        ass_sm(self, wc_sm, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.98, 0.11, 1e9)
        
        
        ass_sm(self, wc_sm, '<AFBl>(Lambdab->Lambda(1520)mumu)', 3, 6, -0.131, 0.031)
        ass_sm(self, wc_sm, '<AFBl>(Lambdab->Lambda(1520)mumu)', 6, 8.68, -0.24, 0.04)
        ass_sm(self, wc_sm, '<AFBl>(Lambdab->Lambda(1520)mumu)', 1, 6, -0.102, 0.028)
        

        ass_sm(self, wc_sm, '<S_1cc>(Lambdab->Lambda(1520)mumu)', 0.1, 3, 0.179, 0.027)
        ass_sm(self, wc_sm, '<S_1cc>(Lambdab->Lambda(1520)mumu)', 3, 6, 0.24, 0.04)
        ass_sm(self, wc_sm, '<S_1cc>(Lambdab->Lambda(1520)mumu)', 6, 8.86, 0.36, 0.05)
        ass_sm(self, wc_sm, '<S_1cc>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.22, 0.04)

        
    def test_lambdablambda1520ll_NP(self):
        # compare to NP values assuming 10% uncertainty on form factors in table 1 of 2005.09602
        ass_sm(self, wc_np_mu, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 3, 6, 1.04, 0.13, 1e9)
        ass_sm(self, wc_np_mu, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 6, 8.68, 2.58, 0.32, 1e9)
        ass_sm(self, wc_np_mu, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.77, 0.10, 1e9)
        
        ass_sm(self, wc_np_mu, '<AFBl>(Lambdab->Lambda(1520)mumu)', 3, 6, -0.059, 0.034)
        ass_sm(self, wc_np_mu, '<AFBl>(Lambdab->Lambda(1520)mumu)', 6, 8.68, -0.166, 0.041)
        ass_sm(self, wc_np_mu, '<AFBl>(Lambdab->Lambda(1520)mumu)', 1, 6, -0.031, 0.032)
        
