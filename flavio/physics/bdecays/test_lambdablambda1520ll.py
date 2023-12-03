import unittest
import numpy as np
import flavio
from flavio.util import get_datapath
from wilson import Wilson

path = get_datapath('flavio', 'data/test/')
sm_array = np.load(path+'2009.09313_digitized.npz')

class TestLambdabLambda1520(unittest.TestCase):

    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.wc_np_mu = Wilson({'C9_bsmumu': -1.11}, scale=4.8, eft='WET', basis='flavio')
        self.ang_coef_List = ['1c', '1cc', '1ss', '2c', '2cc', '2ss', '3ss', '4ss', '5s', '5sc', '6s', '6sc']

    def assert_obs(self, name, target, NP=False, rtol=0.01, scalef=1, **kwargs):
        if NP:
            c = flavio.np_prediction(name, self.wc_np_mu, **kwargs)*scalef
        else:
            c = flavio.sm_prediction(name, **kwargs)*scalef
        self.assertAlmostEqual(c/target, 1, delta=rtol)
        return None

    def assert_sm_arrays(self, name, targetArray, bound, rtol=0.01, atol=0, scalef=1):
        values_to_use = targetArray[0] < 16.79
        q2_vals = targetArray[0][values_to_use]
        target = targetArray[1][values_to_use]
        c = np.array([flavio.sm_prediction(name, q2=q2val)*scalef for q2val in q2_vals])
        if bound == 'central':
            shift = 0
        else:
            unc = np.array([flavio.sm_uncertainty(name, q2=q2val, N=100)*scalef for q2val in q2_vals])
            if bound == 'upper':
                shift = +unc
            elif bound == 'lower':
                shift = -unc
        np.testing.assert_allclose(c+shift, target, rtol=rtol, atol=atol, verbose=True)
        return None


class TestLambdabLambda1520_FF(TestLambdabLambda1520):

    def test_lambdab_lambda1520_ff(self):
        flavio.config['implementation']['Lambdab->Lambda(1520) form factor'] = 'Lambdab->Lambda(1520) LatticeQCD'
        br = flavio.default_parameters.get_central('BR(Lambda(1520)->NKbar)_exp')
        br_inv2 = 2e9/br

        # Comparison to figure 6 in arXiv:2009.09313v2
        # Scale factor BRinv2 used since in np-array factor e^(-9) not included
        dBR = 'dBR/dq2(Lambdab->Lambda(1520)mumu)'
        self.assert_sm_arrays(dBR, sm_array['dB_central'], 'central', scalef=br_inv2, rtol=0.02, atol=0.03)
        self.assert_sm_arrays(dBR, sm_array['dB_upper'], 'upper', scalef=br_inv2, rtol=0.08, atol=0.03)
        self.assert_sm_arrays(dBR, sm_array['dB_lower'], 'lower', scalef=br_inv2, rtol=0.12, atol=0.03)

        # Comparison to figure 9 right in arXiv:2009.09313v2
        Afb = 'AFBl(Lambdab->Lambda(1520)mumu)'
        self.assert_sm_arrays(Afb, sm_array['AFBl_central'], 'central', rtol=0.02, atol=0.03)
        self.assert_sm_arrays(Afb, sm_array['AFBl_upper'], 'upper', rtol=0.08, atol=0.03)
        self.assert_sm_arrays(Afb, sm_array['AFBl_lower'], 'lower', rtol=0.08, atol=0.03)

        # Comparison to figure 7 top right in arXiv:2009.09313v2
        S1cc = 'S_1cc(Lambdab->Lambda(1520)mumu)'
        self.assert_sm_arrays(S1cc, sm_array['S1cc_central'], 'central', rtol=0.02, atol=0.03)
        self.assert_sm_arrays(S1cc, sm_array['S1cc_upper'], 'upper', rtol=0.08, atol=0.03)
        self.assert_sm_arrays(S1cc, sm_array['S1cc_lower'], 'lower', rtol=0.08, atol=0.03)

        flavio.config['implementation']['Lambdab->Lambda(1520) form factor'] = 'Lambdab->Lambda(1520) MCN'
        # Comparison to figure on slide 12 S.Meinel b-baryon FEST 2020
        self.assert_sm_arrays(dBR, sm_array['dB_QM'], 'central', scalef=br_inv2, rtol=0.02, atol=0.03)

        # Comparison to figure on slide 14 S.Meinel b-baryon FEST 2020
        self.assert_sm_arrays(Afb, sm_array['AFBl_QM'], 'central', rtol=0.02, atol=0.03)

        # Comparison to figure on slide 13 S.Meinel b-baryon FEST 2020
        self.assert_sm_arrays(S1cc, sm_array['S1cc_QM'], 'central', rtol=0.02, atol=0.03)


class TestLambdabLambda1520ll(TestLambdabLambda1520):

    def test_lambdablambda1520ll_binned_qmff_sm(self):
        # compare to SM values in table 1 of 2005.09602 using the quark model form factors
        # Differences due to C9eff = C9 and C7eff = C7
        flavio.config['implementation']['Lambdab->Lambda(1520) form factor'] = 'Lambdab->Lambda(1520) MCN'

        self.assert_obs('<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 0.397, rtol=0.18, scalef=1e9, q2min=0.1, q2max=3)
        self.assert_obs('<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 1.29, rtol=0.08, scalef=1e9, q2min=3, q2max=6)
        self.assert_obs('<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 3.22, rtol=0.10, scalef=1e9, q2min=6, q2max=8.68)
        self.assert_obs('<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 0.95, rtol=0.08, scalef=1e9, q2min=1, q2max=6)

        self.assert_obs('<AFBl>(Lambdab->Lambda(1520)mumu)', 0.048, rtol=0.57, q2min=0.1, q2max=3)
        self.assert_obs('<AFBl>(Lambdab->Lambda(1520)mumu)', -0.127, rtol=0.16, q2min=3, q2max=6)
        self.assert_obs('<AFBl>(Lambdab->Lambda(1520)mumu)', -0.235, rtol=0.03, q2min=6, q2max=8.68)
        self.assert_obs('<AFBl>(Lambdab->Lambda(1520)mumu)', -0.098, rtol=0.24, q2min=1, q2max=6)


        self.assert_obs('<S_1cc>(Lambdab->Lambda(1520)mumu)', 0.181, rtol=0.30, q2min=0.1, q2max=3)
        self.assert_obs('<S_1cc>(Lambdab->Lambda(1520)mumu)', 0.242, rtol=0.07, q2min=3, q2max=6)
        self.assert_obs('<S_1cc>(Lambdab->Lambda(1520)mumu)', 0.361, rtol=0.05, q2min=6, q2max=8.68)
        self.assert_obs('<S_1cc>(Lambdab->Lambda(1520)mumu)', 0.221, rtol=0.06, q2min=1, q2max=6)

    def test_lambdablambda1520ll_binned_qmff_np(self):
        # compare to NP values in table 1 of 2005.09602 using quark model form factors
        flavio.config['implementation']['Lambdab->Lambda(1520) form factor'] = 'Lambdab->Lambda(1520) MCN'

        self.assert_obs('<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 0.337, NP=True, rtol=0.19, scalef=1e9, q2min=0.1, q2max=3)
        self.assert_obs('<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 1.04, NP=True, rtol=0.06, scalef=1e9, q2min=3, q2max=6)
        self.assert_obs('<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 2.58, NP=True, rtol=0.07, scalef=1e9, q2min=6, q2max=8.68)
        self.assert_obs('<dBR/dq2>(Lambdab->Lambda(1520)mumu)', 0.77, NP=True, rtol=0.07, scalef=1e9, q2min=1, q2max=6)

        self.assert_obs('<AFBl>(Lambdab->Lambda(1520)mumu)', 0.098  , NP=True, rtol=0.27, q2min=0.1, q2max=3)
        self.assert_obs('<AFBl>(Lambdab->Lambda(1520)mumu)', -0.059, NP=True, rtol=0.39, q2min=3, q2max=6)
        self.assert_obs('<AFBl>(Lambdab->Lambda(1520)mumu)', -0.166, NP=True, rtol=0.01, q2min=6, q2max=8.68)
        self.assert_obs('<AFBl>(Lambdab->Lambda(1520)mumu)', -0.031, NP=True, rtol=0.86, q2min=1, q2max=6)

    def test_lambdablambda1520mm_all_observables(self):
        # Test if all the observables with l=mu work fine.
        names = [
            '<dBR/dq2>(Lambdab->Lambda(1520)mumu)',
            '<AFBh>(Lambdab->Lambda(1520)mumu)',
            '<AFBlh>(Lambdab->Lambda(1520)mumu)',
            '<FL>(Lambdab->Lambda(1520)mumu)',
            '<A_1c>(Lambdab->Lambda(1520)mumu)',
            '<S_1c>(Lambdab->Lambda(1520)mumu)',
            '<A_1cc>(Lambdab->Lambda(1520)mumu)',
            '<S_1cc>(Lambdab->Lambda(1520)mumu)',
            '<A_1ss>(Lambdab->Lambda(1520)mumu)',
            '<S_1ss>(Lambdab->Lambda(1520)mumu)',
            '<A_2c>(Lambdab->Lambda(1520)mumu)',
            '<S_2c>(Lambdab->Lambda(1520)mumu)',
            '<A_2cc>(Lambdab->Lambda(1520)mumu)',
            '<S_2cc>(Lambdab->Lambda(1520)mumu)',
            '<A_2ss>(Lambdab->Lambda(1520)mumu)',
            '<S_2ss>(Lambdab->Lambda(1520)mumu)',
            '<A_3ss>(Lambdab->Lambda(1520)mumu)',
            '<S_3ss>(Lambdab->Lambda(1520)mumu)',
            '<A_4ss>(Lambdab->Lambda(1520)mumu)',
            '<S_4ss>(Lambdab->Lambda(1520)mumu)',
            '<A_5sc>(Lambdab->Lambda(1520)mumu)',
            '<S_5sc>(Lambdab->Lambda(1520)mumu)',
            '<A_6s>(Lambdab->Lambda(1520)mumu)',
            '<S_6s>(Lambdab->Lambda(1520)mumu)',
            '<A_6sc>(Lambdab->Lambda(1520)mumu)',
            '<S_6sc>(Lambdab->Lambda(1520)mumu)',
        ]
        for name in names:
            flavio.sm_prediction(name, q2min=1, q2max=6)

    def test_lambdablambda1520ee_all_observables(self):
        # Test if all the observables with l=e work fine.
        names = [
            '<dBR/dq2>(Lambdab->Lambda(1520)ee)',
            '<AFBh>(Lambdab->Lambda(1520)ee)',
            '<AFBlh>(Lambdab->Lambda(1520)ee)',
            '<FL>(Lambdab->Lambda(1520)ee)',
            '<A_1c>(Lambdab->Lambda(1520)ee)',
            '<S_1c>(Lambdab->Lambda(1520)ee)',
            '<A_1cc>(Lambdab->Lambda(1520)ee)',
            '<S_1cc>(Lambdab->Lambda(1520)ee)',
            '<A_1ss>(Lambdab->Lambda(1520)ee)',
            '<S_1ss>(Lambdab->Lambda(1520)ee)',
            '<A_2c>(Lambdab->Lambda(1520)ee)',
            '<S_2c>(Lambdab->Lambda(1520)ee)',
            '<A_2cc>(Lambdab->Lambda(1520)ee)',
            '<S_2cc>(Lambdab->Lambda(1520)ee)',
            '<A_2ss>(Lambdab->Lambda(1520)ee)',
            '<S_2ss>(Lambdab->Lambda(1520)ee)',
            '<A_3ss>(Lambdab->Lambda(1520)ee)',
            '<S_3ss>(Lambdab->Lambda(1520)ee)',
            '<A_4ss>(Lambdab->Lambda(1520)ee)',
            '<S_4ss>(Lambdab->Lambda(1520)ee)',
            '<A_5sc>(Lambdab->Lambda(1520)ee)',
            '<S_5sc>(Lambdab->Lambda(1520)ee)',
            '<A_6s>(Lambdab->Lambda(1520)ee)',
            '<S_6s>(Lambdab->Lambda(1520)ee)',
            '<A_6sc>(Lambdab->Lambda(1520)ee)',
            '<S_6sc>(Lambdab->Lambda(1520)ee)',
        ]
        for name in names:
            flavio.sm_prediction(name, q2min=1, q2max=6)
