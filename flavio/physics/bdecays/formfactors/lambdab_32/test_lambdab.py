import unittest
import flavio
from flavio.classes import Implementation
from flavio.physics.bdecays.test_lambdablambdall import ass_sm

wc_sm = flavio.WilsonCoefficients()

def pred_sm(s, name, q2val, target, delta, scalef=1):
    obs = flavio.classes.Observable[name]
    c = flavio.sm_prediction(name, q2=q2val)*scalef
    print(c)
    s.assertAlmostEqual(c, target, delta=delta)


class TestLambdabLambda1520_FF(unittest.TestCase):

    def test_lambdab_lambda1520_ff(self):
        flavio.config['implementation']['Lambdab->Lambda(1520) form factor'] = 'Lambdab->Lambda(1520) LatticeQCD'
        par = flavio.default_parameters.get_central_all()
        mLst = par['m_Lambda(1520)']
        mLb = par['m_Lambdab']
        BR = par['BR(Lambda(1520)->NKbar)_exp']
        BRinv = 2/BR

        # Comparison to figure 6 in arXiv:2009.09313
        pred_sm(self, 'dBR/dq2(Lambdab->Lambda(1520)mumu)', 16.0, pow(3.7, -9), pow(0.2, -9), BRinv)
        pred_sm(self, 'dBR/dq2(Lambdab->Lambda(1520)mumu)', 16.4, pow(1.6, -9), pow(0.2, -9), BRinv)
        pred_sm(self, 'dBR/dq2(Lambdab->Lambda(1520)mumu)', 16.8, 0.0, pow(0.01, -9), BRinv)

        # Comparison to figure 9 in arXiv:2009.09313
        pred_sm(self, 'AFBl(Lambdab->Lambda(1520)mumu)', 16.0, -0.13, 0.02)
        pred_sm(self, 'AFBl(Lambdab->Lambda(1520)mumu)', 16.4, 0.0, 0.02)
        pred_sm(self, 'AFBl(Lambdab->Lambda(1520)mumu)', 16.8, 0.32, 0.02)

        # Comparison to figure 7 in arXiv:2009.09313
        pred_sm(self, 'S_1cc(Lambdab->Lambda(1520)mumu)', 16.0, 0.6, 0.02)
        pred_sm(self, 'S_1cc(Lambdab->Lambda(1520)mumu)', 16.4, 0.6, 0.02)
        pred_sm(self, 'S_1cc(Lambdab->Lambda(1520)mumu)', 16.8, 0.44, 0.02)

        
        flavio.config['implementation']['Lambdab->Lambda(1520) form factor'] = 'Lambdab->Lambda(1520) MCN'

        # Comparison to figure on slide 12 S.Meinel b-baryon FEST 2020
        pred_sm(self, 'dBR/dq2(Lambdab->Lambda(1520)mumu)', 16.0, pow(5.4, -9), pow(0.2, -9), BRinv)
        pred_sm(self, 'dBR/dq2(Lambdab->Lambda(1520)mumu)', 16.4, pow(2.2, -9), pow(0.2, -9), BRinv)
        pred_sm(self, 'dBR/dq2(Lambdab->Lambda(1520)mumu)', 16.6, 0.0, pow(0.01, -9), BRinv)

        # Comparison to figure on slide 14 S.Meinel b-baryon FEST 2020
        pred_sm(self, 'AFBl(Lambdab->Lambda(1520)mumu)', 16.0, -0.16, 0.02)
        pred_sm(self, 'AFBl(Lambdab->Lambda(1520)mumu)', 16.4, -0.08, 0.02)
        pred_sm(self, 'AFBl(Lambdab->Lambda(1520)mumu)', 16.6, 0.0, 0.02)
        pred_sm(self, 'AFBl(Lambdab->Lambda(1520)mumu)', 16.8, 0.22, 0.02)

        # Comparison to figure on slide 13 S.Meinel b-baryon FEST 2020
        pred_sm(self, 'S_1cc(Lambdab->Lambda(1520)mumu)', 16.0, 0.6, 0.02)
        pred_sm(self, 'S_1cc(Lambdab->Lambda(1520)mumu)', 16.4, 0.6, 0.02)
        pred_sm(self, 'S_1cc(Lambdab->Lambda(1520)mumu)', 16.8, 0.38, 0.02)

        
