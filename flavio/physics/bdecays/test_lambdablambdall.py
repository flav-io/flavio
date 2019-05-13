import unittest
import numpy as np
import flavio


wc_sm = flavio.WilsonCoefficients()
# choose parameters as required to compare numerics to arXiv:1602.01399
par_nominal = flavio.default_parameters.copy()
flavio.physics.bdecays.formfactors.lambdab_12.lattice_parameters.lattice_load_nominal(par_nominal)
par_nominal.set_constraint('Vcb', 0.04175)
par_nominal.set_constraint('tau_Lambdab', 1/4.49e-13) # PDG 2016 value
par_nominal.set_constraint('Lambda->ppi alpha_-', 0.642) # PDG 2016 value
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
        ass_sm(self, '<dBR/dq2>(Lambdab->Lambdamumu)', 15, 20, 0.756, 0.003, 1e7)
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
