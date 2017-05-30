import unittest
import flavio
from flavio.classes import Observable, Prediction
from flavio.functions import get_dependent_parameters_sm

class TestFunctions(unittest.TestCase):
    def test_functions(self):
        o = Observable('test_obs')
        o.arguments = ['x']
        def f(wc_obj, par_dict, x):
            return x
        pr  = Prediction('test_obs', f )
        wc_obj = None
        self.assertEqual(flavio.sm_prediction('test_obs', 7), 7)
        self.assertEqual(flavio.np_prediction('test_obs', x=7, wc_obj=wc_obj), 7)
        self.assertEqual(flavio.sm_uncertainty('test_obs', 7), 0)
        self.assertEqual(flavio.np_uncertainty('test_obs', x=7, wc_obj=wc_obj), 0)
        self.assertEqual(flavio.sm_uncertainty('test_obs', 7, threads=2), 0)
        self.assertEqual(flavio.np_uncertainty('test_obs', x=7, wc_obj=wc_obj, threads=2), 0)
        # delete dummy instance
        Observable.del_instance('test_obs')

    def test_get_dep_par(self):
        self.assertEqual(
            get_dependent_parameters_sm('BR(Bs->mumu)'),
            {'DeltaGamma/Gamma_Bs', 'GF', 'Vcb', 'Vub', 'Vus', 'alpha_e', 'alpha_s', 'f_Bs', 'gamma', 'm_Bs', 'm_b', 'm_mu', 'm_s', 'tau_Bs', 'm_t'}
        )
        self.assertEqual(
            get_dependent_parameters_sm('BR(B0->ee)'),
            {'GF', 'Vcb', 'Vub', 'Vus', 'alpha_e', 'alpha_s', 'f_B0', 'gamma', 'm_B0', 'm_b', 'm_e', 'm_d', 'tau_B0', 'm_t'}
        )
        # for more complicated cases, just check there is no error
        get_dependent_parameters_sm('dBR/dq2(B+->Kmumu)', 3)
        get_dependent_parameters_sm('<dBR/dq2>(B+->Kmumu)', 3, 5)
        get_dependent_parameters_sm('dBR/dq2(B+->Kmumu)', q2=3)
        get_dependent_parameters_sm('<dBR/dq2>(B+->Kmumu)', q2min=3, q2max=5)
