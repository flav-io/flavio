import unittest
import os
import flavio
from flavio.classes import Implementation
from flavio.physics.bdecays.test_lambdablambdall import ass_sm
from flavio.util import get_datapath
import numpy as np

path = get_datapath('flavio', 'data/test/')
sm_array = np.load(path+'2009.09313_digitized.npz')

wc_sm = flavio.WilsonCoefficients()

def pred_sm(s, name, q2val, target, delta, scalef=1):
    obs = flavio.classes.Observable[name]
    c = flavio.sm_prediction(name, q2=q2val)*scalef
    s.assertAlmostEqual(c, target, delta=delta)
    return None


def pred_sm_arrays(s, name, targetArray, Unc, delta, scalef=1):
    obs = flavio.classes.Observable[name]
    for i, q2val in enumerate(targetArray[0]):
        target = targetArray[1][i]
        c = flavio.sm_prediction(name, q2=q2val)*scalef
        if Unc == True:
            u = flavio.sm_uncertainty(name, q2=q2val, N=100)*scalef
            if target > c :
                c = c + u
            else :
                c = c - u
        s.assertAlmostEqual(c, target, delta=delta)
    return None


class TestLambdabLambda1520_FF(unittest.TestCase):

    def test_lambdab_lambda1520_ff(self):
        flavio.config['implementation']['Lambdab->Lambda(1520) form factor'] = 'Lambdab->Lambda(1520) LatticeQCD'
        par = flavio.default_parameters.get_central_all()
        mLst = par['m_Lambda(1520)']
        mLb = par['m_Lambdab']
        br = par['BR(Lambda(1520)->NKbar)_exp']
        br_inv = 2/br
        br_inv2 = 2e9/br
        
        # Comparison to figure 6 in arXiv:2009.09313v2
        # Scale factor BRinv2 used since in np-array factor e^(-9) not included
        Dbr = 'dBR/dq2(Lambdab->Lambda(1520)mumu)'
        
        pred_sm_arrays(self, Dbr, sm_array['dB_central'], False, 0.05, br_inv2)
        pred_sm_arrays(self, Dbr, sm_array['dB_upper'], True, 0.15, br_inv2)
        pred_sm_arrays(self, Dbr, sm_array['dB_lower'], True, 0.2, br_inv2)

        # BRinv since not included in figure
        pred_sm(self, Dbr, 16.0, 2.5e-9, 0.1e-9, br_inv)
        pred_sm(self, Dbr, 16.4, 1.2e-9, 0.1e-9, br_inv)
        pred_sm(self, Dbr, 16.8, 0.0, 0.05e-9, br_inv)
        
        
        # Comparison to figure 9 right in arXiv:2009.09313v2
        Afb = 'AFBl(Lambdab->Lambda(1520)mumu)'
        
        pred_sm_arrays(self, Afb, sm_array['AFBl_central'], False, 0.06)
        pred_sm_arrays(self, Afb, sm_array['AFBl_upper'], True, 0.11)
        pred_sm_arrays(self, Afb, sm_array['AFBl_lower'], True, 0.11)
        
        pred_sm(self, Afb, 16.0, -0.09, 0.01)
        pred_sm(self, Afb, 16.4, 0.05, 0.01)
        pred_sm(self, Afb, 16.8, 0.25, 0.02)
        
        # Comparison to figure 7 top right in arXiv:2009.09313v3
        S1cc = 'S_1cc(Lambdab->Lambda(1520)mumu)'

        pred_sm_arrays(self, S1cc, sm_array['S1cc_central'], False, 0.01)
        pred_sm_arrays(self, S1cc, sm_array['S1cc_upper'], True, 0.1)
        pred_sm_arrays(self, S1cc, sm_array['S1cc_lower'], True, 0.1)
        
        pred_sm(self, S1cc, 16.0, 0.56, 0.01)
        pred_sm(self, S1cc, 16.4, 0.56, 0.02)
        pred_sm(self, S1cc, 16.8, 0.36, 0.02)
        
        flavio.config['implementation']['Lambdab->Lambda(1520) form factor'] = 'Lambdab->Lambda(1520) MCN'
        
        # Comparison to figure on slide 12 S.Meinel b-baryon FEST 2020
        pred_sm_arrays(self, Dbr, sm_array['dB_QM'], False, 0.07, br_inv2)
        
        pred_sm(self, Dbr, 16.0, 5.4e-9, 0.1e-9, br_inv)
        pred_sm(self, Dbr, 16.4, 2.2e-9, 0.1e-9, br_inv)
        pred_sm(self, Dbr, 16.6, 0.8e-9, 0.1e-9, br_inv)
        
        # Comparison to figure on slide 14 S.Meinel b-baryon FEST 2020
        pred_sm_arrays(self, Afb, sm_array['AFBl_QM'], False, 0.07)

        pred_sm(self, Afb, 16.0, -0.16, 0.02)
        pred_sm(self, Afb, 16.4, -0.08, 0.02)
        pred_sm(self, Afb, 16.6, 0.0, 0.02)
        pred_sm(self, Afb, 16.8, 0.22, 0.02)

        # Comparison to figure on slide 13 S.Meinel b-baryon FEST 2020
        pred_sm_arrays(self, S1cc, sm_array['S1cc_QM'], False, 0.02)
        
        pred_sm(self, S1cc, 16.0, 0.6, 0.02)
        pred_sm(self, S1cc, 16.4, 0.6, 0.02)
        pred_sm(self, S1cc, 16.8, 0.38, 0.02)


        
