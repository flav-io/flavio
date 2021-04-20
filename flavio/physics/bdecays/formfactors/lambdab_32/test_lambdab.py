import unittest
import flavio
from flavio.classes import Implementation
from flavio.physics.bdecays.test_lambdablambdall import ass_sm

wc_sm = flavio.WilsonCoefficients()

def pred_sm(s, name, q2val, target, delta, scalef=1):
    obs = flavio.classes.Observable[name]
    c = flavio.sm_prediction(name, q2=q2val)*scalef
    s.assertAlmostEqual(c, target, delta=delta)


class TestLambdabLambda1520_FF(unittest.TestCase):

    def test_lambdab_lambda1520_ff(self):
        flavio.config['implementation']['Lambdab->Lambda(1520) form factor'] = 'Lambdab->Lambda(1520) LatticeQCD'
        par = flavio.default_parameters.get_central_all()
        mLst = par['m_Lambda(1520)']
        mLb = par['m_Lambdab']
        BR = par['BR(Lambda(1520)->NKbar)_exp']
        BRinv = 2/BR

        # Comparison to figure 6 in arXiv:2009.09313v2
        pred_sm(self, 'dBR/dq2(Lambdab->Lambda(1520)mumu)', 16.0, 2.5e-9, 0.1e-9, BRinv)
        pred_sm(self, 'dBR/dq2(Lambdab->Lambda(1520)mumu)', 16.4, 1.2e-9, 0.1e-9, BRinv)
        pred_sm(self, 'dBR/dq2(Lambdab->Lambda(1520)mumu)', 16.8, 0.0, 0.05e-9, BRinv)

        # Comparison to figure 9 right in arXiv:2009.09313v2
        pred_sm(self, 'AFBl(Lambdab->Lambda(1520)mumu)', 16.0, -0.09, 0.01)
        pred_sm(self, 'AFBl(Lambdab->Lambda(1520)mumu)', 16.4, 0.05, 0.01)
        pred_sm(self, 'AFBl(Lambdab->Lambda(1520)mumu)', 16.8, 0.25, 0.02)

        # Comparison to figure 7 top right in arXiv:2009.09313v3
        pred_sm(self, 'S_1cc(Lambdab->Lambda(1520)mumu)', 16.0, 0.56, 0.01)
        pred_sm(self, 'S_1cc(Lambdab->Lambda(1520)mumu)', 16.4, 0.56, 0.02)
        pred_sm(self, 'S_1cc(Lambdab->Lambda(1520)mumu)', 16.8, 0.36, 0.02)

        
        flavio.config['implementation']['Lambdab->Lambda(1520) form factor'] = 'Lambdab->Lambda(1520) MCN'

        # Comparison to figure on slide 12 S.Meinel b-baryon FEST 2020
        pred_sm(self, 'dBR/dq2(Lambdab->Lambda(1520)mumu)', 16.0, 5.4e-9, 0.1e-9, BRinv)
        pred_sm(self, 'dBR/dq2(Lambdab->Lambda(1520)mumu)', 16.4, 2.2e-9, 0.1e-9, BRinv)
        pred_sm(self, 'dBR/dq2(Lambdab->Lambda(1520)mumu)', 16.6, 0.8e-9, 0.1e-9, BRinv)

        # Comparison to figure on slide 14 S.Meinel b-baryon FEST 2020
        pred_sm(self, 'AFBl(Lambdab->Lambda(1520)mumu)', 16.0, -0.16, 0.02)
        pred_sm(self, 'AFBl(Lambdab->Lambda(1520)mumu)', 16.4, -0.08, 0.02)
        pred_sm(self, 'AFBl(Lambdab->Lambda(1520)mumu)', 16.6, 0.0, 0.02)
        pred_sm(self, 'AFBl(Lambdab->Lambda(1520)mumu)', 16.8, 0.22, 0.02)

        # Comparison to figure on slide 13 S.Meinel b-baryon FEST 2020
        pred_sm(self, 'S_1cc(Lambdab->Lambda(1520)mumu)', 16.0, 0.6, 0.02)
        pred_sm(self, 'S_1cc(Lambdab->Lambda(1520)mumu)', 16.4, 0.6, 0.02)
        pred_sm(self, 'S_1cc(Lambdab->Lambda(1520)mumu)', 16.8, 0.38, 0.02)

        
