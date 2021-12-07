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


ang_coef_List = ['1c', '1cc', '1ss', '2c', '2cc', '2ss', '3ss', '4ss', '5s', '5sc', '6s', '6sc']

def ass_sm(s, wc, name, q2min, q2max, target, delta, scalef=1):
    obs = flavio.classes.Observable[name]
    c = obs.prediction_central(par_nominal, wc, q2min, q2max)*scalef
    s.assertAlmostEqual(c, target, delta=delta)
    
    
class TestLambdabLambda1520ll(unittest.TestCase):
    def test_lambdablambda1520ll_binned_qmff_sm(self):
        # compare to SM values assuming 10% uncertainty on form factors in table 1 of 2005.09602 using the quark model form factors
        # Differences due to C9eff = C9 and C7eff = C7
        # Second test of angular distribution using lattice QCD form factors
        # in /bdecays/formfactors/lambdab_32/test_lambdab.py 
        ass_sm(self, wc_sm, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 0.1, 3, 0.42, 0.05, 1e9)
        ass_sm(self, wc_sm, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 3, 6, 1.34, 0.16, 1e9)
        ass_sm(self, wc_sm, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 6, 8.86, 3.4, 0.4, 1e9)
        ass_sm(self, wc_sm, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.98, 0.11, 1e9)
        
        ass_sm(self, wc_sm, '<AFBl>(Lambdab->Lambda(1520)mumu)', 3, 6, -0.131, 0.031)
        ass_sm(self, wc_sm, '<AFBl>(Lambdab->Lambda(1520)mumu)', 6, 8.86, -0.24, 0.04)
        ass_sm(self, wc_sm, '<AFBl>(Lambdab->Lambda(1520)mumu)', 1, 6, -0.102, 0.028)
        

        ass_sm(self, wc_sm, '<S_1cc>(Lambdab->Lambda(1520)mumu)', 0.1, 3, 0.179, 0.054)
        ass_sm(self, wc_sm, '<S_1cc>(Lambdab->Lambda(1520)mumu)', 3, 6, 0.24, 0.04)
        ass_sm(self, wc_sm, '<S_1cc>(Lambdab->Lambda(1520)mumu)', 6, 8.86, 0.36, 0.05)
        ass_sm(self, wc_sm, '<S_1cc>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.22, 0.04)
        
    def test_lambdablambda1520ll_binned_qmff_np(self):
        # compare to NP values assuming 10% uncertainty on form factors in table 1 of 2005.09602 using quark model form factors
        ass_sm(self, wc_np_mu, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 3, 6, 1.04, 0.13, 1e9)
        ass_sm(self, wc_np_mu, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 6, 8.86, 2.58, 0.32, 1e9)
        ass_sm(self, wc_np_mu, '<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.77, 0.10, 1e9)
        
        ass_sm(self, wc_np_mu, '<AFBl>(Lambdab->Lambda(1520)mumu)', 3, 6, -0.059, 0.034)
        ass_sm(self, wc_np_mu, '<AFBl>(Lambdab->Lambda(1520)mumu)', 6, 8.86, -0.166, 0.041)
        ass_sm(self, wc_np_mu, '<AFBl>(Lambdab->Lambda(1520)mumu)', 1, 6, -0.031, 0.032)

    def test_lambdablambda1520mm_all_observables(self):
        # Test if all the observables with l=mu work fine. 
        ass_sm(self, wc_sm, '<AFBh>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.0, 0.0000001)
        ass_sm(self, wc_sm, '<AFBlh>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.0, 0.0000001)
        ass_sm(self, wc_sm, '<FL>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.8, 0.1) 
        ass_sm(self, wc_sm, '<A_1c>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<S_1c>(Lambdab->Lambda(1520)mumu)', 1, 6, -0.1, 0.1)
        ass_sm(self, wc_sm, '<A_1cc>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<S_1cc>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.2, 0.1)
        ass_sm(self, wc_sm, '<A_1ss>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<S_1ss>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.9, 0.1)
        ass_sm(self, wc_sm, '<A_2c>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<S_2c>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<A_2cc>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<S_2cc>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<A_2ss>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<S_2ss>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.2, 0.1)
        ass_sm(self, wc_sm, '<A_3ss>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<S_3ss>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<A_4ss>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<S_4ss>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<A_5sc>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<S_5sc>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<A_6s>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.0, 0.01)
        ass_sm(self, wc_sm, '<S_6s>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.0, 0.01)
        ass_sm(self, wc_sm, '<A_6sc>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<S_6sc>(Lambdab->Lambda(1520)mumu)', 1, 6, 0.0, 0.1)

    def test_lambdablambda1520mm_all_observables(self):
        # Test if all the observables with l=e work fine. 
        ass_sm(self, wc_sm, '<dBR/dq2>(Lambdab->Lambda(1520)ee)', 1, 6, 0.98, 0.11, 1e9)
        ass_sm(self, wc_sm, '<AFBl>(Lambdab->Lambda(1520)ee)', 1, 6, -0.102, 0.028)
        ass_sm(self, wc_sm, '<AFBh>(Lambdab->Lambda(1520)ee)', 1, 6, 0.0, 0.0000001)
        ass_sm(self, wc_sm, '<AFBlh>(Lambdab->Lambda(1520)ee)', 1, 6, 0.0, 0.0000001) 
        ass_sm(self, wc_sm, '<FL>(Lambdab->Lambda(1520)ee)', 1, 6, 0.8, 0.1) 
        ass_sm(self, wc_sm, '<A_1c>(Lambdab->Lambda(1520)ee)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<S_1c>(Lambdab->Lambda(1520)ee)', 1, 6, -0.1, 0.1)
        ass_sm(self, wc_sm, '<A_1cc>(Lambdab->Lambda(1520)ee)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<S_1cc>(Lambdab->Lambda(1520)ee)', 1, 6, 0.2, 0.1)
        ass_sm(self, wc_sm, '<A_1ss>(Lambdab->Lambda(1520)ee)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<S_1ss>(Lambdab->Lambda(1520)ee)', 1, 6, 0.9, 0.1)
        ass_sm(self, wc_sm, '<A_2c>(Lambdab->Lambda(1520)ee)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<S_2c>(Lambdab->Lambda(1520)ee)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<A_2cc>(Lambdab->Lambda(1520)ee)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<S_2cc>(Lambdab->Lambda(1520)ee)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<A_2ss>(Lambdab->Lambda(1520)ee)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<S_2ss>(Lambdab->Lambda(1520)ee)', 1, 6, 0.2, 0.1)
        ass_sm(self, wc_sm, '<A_3ss>(Lambdab->Lambda(1520)ee)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<S_3ss>(Lambdab->Lambda(1520)ee)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<A_4ss>(Lambdab->Lambda(1520)ee)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<S_4ss>(Lambdab->Lambda(1520)ee)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<A_5sc>(Lambdab->Lambda(1520)ee)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<S_5sc>(Lambdab->Lambda(1520)ee)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<A_6s>(Lambdab->Lambda(1520)ee)', 1, 6, 0.0, 0.01)
        ass_sm(self, wc_sm, '<S_6s>(Lambdab->Lambda(1520)ee)', 1, 6, 0.0, 0.01)
        ass_sm(self, wc_sm, '<A_6sc>(Lambdab->Lambda(1520)ee)', 1, 6, 0.0, 0.1)
        ass_sm(self, wc_sm, '<S_6sc>(Lambdab->Lambda(1520)ee)', 1, 6, 0.0, 0.1)



 



        

        
