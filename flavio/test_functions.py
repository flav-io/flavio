import unittest
import numpy.testing as npt
import flavio
from flavio.classes import Observable, Prediction
from flavio.functions import get_dependent_parameters_sm
import copy
import numpy as np

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

    def test_sm_covariance(self):
        o1 = Observable( 'test_obs 1' )
        o2 = Observable( 'test_obs 2' )
        def f1(wc_obj, par_dict):
            return par_dict['m_b']
        def f2(wc_obj, par_dict):
            return par_dict['m_c']
        Prediction('test_obs 1', f1)
        Prediction('test_obs 2', f2)
        cov_par = np.array([[0.1**2, 0.1*0.2*0.3], [0.1*0.2*0.3, 0.2**2]])
        d = flavio.statistics.probability.MultivariateNormalDistribution([4.2, 1.2], covariance=cov_par)
        par = copy.deepcopy(flavio.parameters.default_parameters)
        par.add_constraint(['m_b', 'm_c'], d)
        # test serial
        np.random.seed(135)
        cov = flavio.sm_covariance(['test_obs 1', 'test_obs 2'],
                                   N=1000, par_vary='all', par_obj=par)
        npt.assert_array_almost_equal(cov, cov_par, decimal=3)
        # test parallel
        np.random.seed(135)
        cov_parallel = flavio.sm_covariance(['test_obs 1', 'test_obs 2'],
                                   N=1000, par_vary='all', par_obj=par,
                                   threads=4)
        npt.assert_array_equal(cov, cov_parallel)
        np.random.seed(135)
        cov_1 = flavio.sm_covariance(['test_obs 1'],
                                   N=1000, par_vary='all', par_obj=par)
        # test with single observable
        npt.assert_array_almost_equal(cov_1, cov[0, 0])
        # test with fixed parameter
        cov_f = flavio.sm_covariance(['test_obs 1', 'test_obs 2'],
                                   N=1000, par_vary=['m_b'], par_obj=par)
        npt.assert_array_almost_equal(cov_f, [[cov_par[0, 0], 0], [0, 0]], decimal=3)
        # delete dummy instances
        Observable.del_instance('test_obs 1')
        Observable.del_instance('test_obs 2')
