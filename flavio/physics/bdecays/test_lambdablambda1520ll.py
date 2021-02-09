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
        # Differences due to different C9eff
        # In paper C9eff = C9 + Y , ignoring Yu and DeltaC9
        # But implemented in flavio
        # Better: look at test of lattice QCD and QM form factors 
        ass_sm(self, wc_sm, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 3, 6, 1.29, 0.18, 1e9)
        ass_sm(self, wc_sm, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 6, 8.68, 3.22, 0.42, 1e9)
        ass_sm(self, wc_sm, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.95, 0.13, 1e9)
        
        ass_sm(self, wc_sm, '<AFBl>(Lambdab->Lambda(1520)mumu)', 3, 6, -0.127, 0.033)
        ass_sm(self, wc_sm, '<AFBl>(Lambdab->Lambda(1520)mumu)', 6, 8.68, -0.235, 0.040)
        ass_sm(self, wc_sm, '<AFBl>(Lambdab->Lambda(1520)mumu)', 1, 6, -0.098, 0.031)
        

        
    def test_lambdablambda1520ll_NP(self):
        # compare to NP values assuming 10% uncertainty on form factors in table 1 of 2005.09602
        ass_sm(self, wc_np_mu, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 3, 6, 1.04, 0.13, 1e9)
        ass_sm(self, wc_np_mu, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 6, 8.68, 2.58, 0.32, 1e9)
        ass_sm(self, wc_np_mu, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.77, 0.10, 1e9)
        
        ass_sm(self, wc_np_mu, '<AFBl>(Lambdab->Lambda(1520)mumu)', 3, 6, -0.059, 0.034)
        ass_sm(self, wc_np_mu, '<AFBl>(Lambdab->Lambda(1520)mumu)', 6, 8.68, -0.166, 0.041)
        ass_sm(self, wc_np_mu, '<AFBl>(Lambdab->Lambda(1520)mumu)', 1, 6, -0.031, 0.032)
        
