import unittest
import numpy as np
import flavio


wc_sm = flavio.WilsonCoefficients()
# choose parameters as required to compare numerics to arXiv:1602.01399
par_nominal = flavio.default_parameters.copy()
flavio.physics.bdecays.formfactors.lambdab_12.lattice_parameters.lattice_load_nominal(par_nominal)
par_nominal.set_constraint('Vcb', 0.04175)
par_nominal.set_constraint('gamma', 1.30)
par_nominal.set_constraint("m_t", 173.21)
par_nominal.set_constraint('tau_Lambdab', 1/4.49e-13) # PDG 2016 value
par_nominal.set_constraint('Lambda->ppi alpha_-', 0.642) # PDG 2016 value
par_nominal.set_constraint('Lambdab polarisation', 1.00) # Used in theory predictions
par_dict = par_nominal.get_central_all()

def ass_sm(s, name, q2min, q2max, target, delta, scalef=1):
    obs = flavio.classes.Observable[name]
    c = obs.prediction_central(par_nominal, wc_sm, q2min, q2max)*scalef
    s.assertAlmostEqual(c, target, delta=delta)

class TestLambdabLambdall(unittest.TestCase):
    def test_lambdablambdall(self):
        # first, make sure we use the same CKM factor as in arXiv:1602.01399 eq. (69)
        self.assertAlmostEqual(abs(flavio.physics.ckm.xi('t', 'bs')(par_dict)), 0.04088, delta=0.0001)
        # compare to table VII of 1602.01399
        ass_sm(self, '<dBR/dq2>(Lambdab->Lambdamumu)', 0.1, 2, 0.25, 0.01, 1e7)
        ass_sm(self, '<dBR/dq2>(Lambdab->Lambdamumu)', 2, 4, 0.18, 0.005, 1e7)
        ass_sm(self, '<dBR/dq2>(Lambdab->Lambdamumu)', 15, 20, 0.756, 0.004, 1e7)
        ass_sm(self, '<dBR/dq2>(Lambdab->Lambdamumu)', 18, 20, 0.665, 0.002, 1e7)
        ass_sm(self, '<FL>(Lambdab->Lambdamumu)', 4, 6, 0.808, 0.007)
        ass_sm(self, '<FL>(Lambdab->Lambdamumu)', 15, 20, 0.409, 0.002)
        ass_sm(self, '<AFBl>(Lambdab->Lambdamumu)', 4, 6, -0.062, 0.005)
        ass_sm(self, '<AFBl>(Lambdab->Lambdamumu)', 15, 20, -0.350, 0.002)
        ass_sm(self, '<AFBh>(Lambdab->Lambdamumu)', 4, 6, -0.311, 0.005)
        ass_sm(self, '<AFBh>(Lambdab->Lambdamumu)', 15, 20, -0.2710, 0.002)
        ass_sm(self, '<AFBlh>(Lambdab->Lambdamumu)', 4, 6, 0.021, 0.005)
        ass_sm(self, '<AFBlh>(Lambdab->Lambdamumu)', 15, 20, 0.1398, 0.002)

    def test_lambdablambdall_subleading(self):
        ta_high = flavio.classes.AuxiliaryQuantity(
        'Lambdab->Lambdall subleading effects at high q2'
        ).prediction_central(par_nominal, wc_sm, q2=15, cp_conjugate=False)
        ta_low = flavio.classes.AuxiliaryQuantity(
        'Lambdab->Lambdall subleading effects at low q2'
        ).prediction_central(par_nominal, wc_sm, q2=1, cp_conjugate=False)
        # check that the keys contain all the transversity amps
        allamps = {('para0','L'), ('para1','L'), ('perp0','L'), ('perp1','L'),
                   ('para0','R'), ('para1','R'), ('perp0','R'), ('perp1','R')}
        self.assertEqual(set(ta_high.keys()), allamps)
        self.assertEqual(set(ta_low.keys()), allamps)
        # check that the central values are actually all zero
        # self.assertEqual(set(ta_high.values()), {0})
        # self.assertEqual(set(ta_low.values()), {0})

    def test_lambdalambdall_mumu_angular_1to6(self):
        # Compare to table 2 of arXiv:1710.00746.
        # These predictions were obtained with eos in 2017 and are slightly
        # different from the ones obtained with flavio, likely due to
        # different input parameters. The tolerances are set accordingly.
        ass_sm(self, '<K1ss>(Lambdab->Lambdamumu)', 1, 6,  0.459, 0.003)
        ass_sm(self, '<K1cc>(Lambdab->Lambdamumu)', 1, 6,  0.081, 0.007)
        ass_sm(self, '<K1c>(Lambdab->Lambdamumu)',  1, 6, -0.005, 0.017)
        ass_sm(self, '<K2ss>(Lambdab->Lambdamumu)', 1, 6, -0.280, 0.002)
        ass_sm(self, '<K2cc>(Lambdab->Lambdamumu)', 1, 6, -0.045, 0.011)
        ass_sm(self, '<K2c>(Lambdab->Lambdamumu)',  1, 6,  0.000, 0.007)
        ass_sm(self, '<K4sc>(Lambdab->Lambdamumu)', 1, 6, -0.025, 0.033)
        ass_sm(self, '<K4s>(Lambdab->Lambdamumu)',  1, 6, -0.003, 0.030)
        ass_sm(self, '<K3sc>(Lambdab->Lambdamumu)', 1, 6,  0.002, 0.002)
        ass_sm(self, '<K3s>(Lambdab->Lambdamumu)',  1, 6,  0.002, 0.003)
        ass_sm(self, '<K11>(Lambdab->Lambdamumu)',  1, 6, -0.366, 0.014)
        ass_sm(self, '<K12>(Lambdab->Lambdamumu)',  1, 6,  0.071, 0.015)
        ass_sm(self, '<K13>(Lambdab->Lambdamumu)',  1, 6,  0.001, 0.010)
        ass_sm(self, '<K14>(Lambdab->Lambdamumu)',  1, 6,  0.243, 0.007)
        ass_sm(self, '<K15>(Lambdab->Lambdamumu)',  1, 6, -0.052, 0.005)
        ass_sm(self, '<K16>(Lambdab->Lambdamumu)',  1, 6,  0.003, 0.011)
        ass_sm(self, '<K17>(Lambdab->Lambdamumu)',  1, 6,  0.004, 0.031)
        ass_sm(self, '<K18>(Lambdab->Lambdamumu)',  1, 6,  0.029, 0.035)
        ass_sm(self, '<K19>(Lambdab->Lambdamumu)',  1, 6, -0.001, 0.002)
        ass_sm(self, '<K20>(Lambdab->Lambdamumu)',  1, 6, -0.003, 0.003)
        ass_sm(self, '<K21>(Lambdab->Lambdamumu)',  1, 6,  0.002, 0.003)
        ass_sm(self, '<K22>(Lambdab->Lambdamumu)',  1, 6, -0.005, 0.003)
        ass_sm(self, '<K23>(Lambdab->Lambdamumu)',  1, 6, -0.147, 0.013)
        ass_sm(self, '<K24>(Lambdab->Lambdamumu)',  1, 6,  0.132, 0.018)
        ass_sm(self, '<K25>(Lambdab->Lambdamumu)',  1, 6, -0.001, 0.003)
        ass_sm(self, '<K26>(Lambdab->Lambdamumu)',  1, 6,  0.004, 0.003)
        ass_sm(self, '<K27>(Lambdab->Lambdamumu)',  1, 6,  0.089, 0.003)
        ass_sm(self, '<K28>(Lambdab->Lambdamumu)',  1, 6, -0.089, 0.021)
        ass_sm(self, '<K29>(Lambdab->Lambdamumu)',  1, 6,  0.000, 0.003)
        ass_sm(self, '<K30>(Lambdab->Lambdamumu)',  1, 6,  0.000, 0.004)
        ass_sm(self, '<K31>(Lambdab->Lambdamumu)',  1, 6,  0.000, 0.001)
        ass_sm(self, '<K32>(Lambdab->Lambdamumu)',  1, 6,  0.075, 0.001)
        ass_sm(self, '<K33>(Lambdab->Lambdamumu)',  1, 6,  0.007, 0.004)
        ass_sm(self, '<K34>(Lambdab->Lambdamumu)',  1, 6,  0.000, 0.001)

    def test_lambdalambdall_mumu_angular_15to20(self):
        # Compare to table 3 of arXiv:1710.00746.
        # These predictions were obtained with eos in 2017 and are slightly
        # different from the ones obtained with flavio, likely due to
        # different input parameters. The tolerances are set accordingly.
        ass_sm(self, '<K1ss>(Lambdab->Lambdamumu)', 15, 20,  0.351, 0.002)
        ass_sm(self, '<K1cc>(Lambdab->Lambdamumu)', 15, 20,  0.298, 0.003)
        ass_sm(self, '<K1c>(Lambdab->Lambdamumu)',  15, 20, -0.236, 0.004)
        ass_sm(self, '<K2ss>(Lambdab->Lambdamumu)', 15, 20, -0.195, 0.001)
        ass_sm(self, '<K2cc>(Lambdab->Lambdamumu)', 15, 20, -0.154, 0.002)
        ass_sm(self, '<K2c>(Lambdab->Lambdamumu)',  15, 20,  0.187, 0.002)
        ass_sm(self, '<K4sc>(Lambdab->Lambdamumu)', 15, 20, -0.022, 0.003)
        ass_sm(self, '<K4s>(Lambdab->Lambdamumu)',  15, 20, -0.100, 0.004)
        ass_sm(self, '<K3sc>(Lambdab->Lambdamumu)', 15, 20,  0.000, 0.001)
        ass_sm(self, '<K3s>(Lambdab->Lambdamumu)',  15, 20, -0.001, 0.001)
        ass_sm(self, '<K11>(Lambdab->Lambdamumu)',  15, 20, -0.064, 0.002)
        ass_sm(self, '<K12>(Lambdab->Lambdamumu)',  15, 20,  0.240, 0.003)
        ass_sm(self, '<K13>(Lambdab->Lambdamumu)',  15, 20, -0.292, 0.003)
        ass_sm(self, '<K14>(Lambdab->Lambdamumu)',  15, 20,  0.034, 0.003)
        ass_sm(self, '<K15>(Lambdab->Lambdamumu)',  15, 20, -0.191, 0.002)
        ass_sm(self, '<K16>(Lambdab->Lambdamumu)',  15, 20,  0.151, 0.002)
        ass_sm(self, '<K17>(Lambdab->Lambdamumu)',  15, 20,  0.102, 0.003)
        ass_sm(self, '<K18>(Lambdab->Lambdamumu)',  15, 20,  0.021, 0.002)
        ass_sm(self, '<K19>(Lambdab->Lambdamumu)',  15, 20,  0.000, 0.001)
        ass_sm(self, '<K20>(Lambdab->Lambdamumu)',  15, 20, -0.001, 0.001)
        ass_sm(self, '<K21>(Lambdab->Lambdamumu)',  15, 20,  0.000, 0.001)
        ass_sm(self, '<K22>(Lambdab->Lambdamumu)',  15, 20, -0.002, 0.002)
        ass_sm(self, '<K23>(Lambdab->Lambdamumu)',  15, 20, -0.299, 0.003)
        ass_sm(self, '<K24>(Lambdab->Lambdamumu)',  15, 20,  0.337, 0.003)
        ass_sm(self, '<K25>(Lambdab->Lambdamumu)',  15, 20, -0.001, 0.002)
        ass_sm(self, '<K26>(Lambdab->Lambdamumu)',  15, 20,  0.001, 0.001)
        ass_sm(self, '<K27>(Lambdab->Lambdamumu)',  15, 20,  0.221, 0.001)
        ass_sm(self, '<K28>(Lambdab->Lambdamumu)',  15, 20, -0.187, 0.001)
        ass_sm(self, '<K29>(Lambdab->Lambdamumu)',  15, 20,  0.000, 0.001)
        ass_sm(self, '<K30>(Lambdab->Lambdamumu)',  15, 20, -0.001, 0.002)
        ass_sm(self, '<K31>(Lambdab->Lambdamumu)',  15, 20,  0.000, 0.001)
        ass_sm(self, '<K32>(Lambdab->Lambdamumu)',  15, 20, -0.046, 0.004)
        ass_sm(self, '<K33>(Lambdab->Lambdamumu)',  15, 20, -0.053, 0.001)
        ass_sm(self, '<K34>(Lambdab->Lambdamumu)',  15, 20,  0.000, 0.001)
