import unittest
import numpy as np
import flavio
from .classes import *
from flavio.classes import *
from flavio.config import config
import copy

class TestClasses(unittest.TestCase):
    def test_fit_class(self):
        o = Observable( 'test_obs' )
        d = NormalDistribution(4.2, 0.2)
        m = Measurement( 'measurement of test_obs' )
        m.add_constraint(['test_obs'], d)
        par = copy.deepcopy(flavio.parameters.default_parameters)
        par.set_constraint('m_b', '4.2+-0.2')
        par.set_constraint('m_c', '1.2+-0.1')
        wc = flavio.physics.eft.WilsonCoefficients()
        with self.assertRaises(AssertionError):
            # same parameter as fit and nuisance
            Fit('test_fit_1', par, wc, ['m_b'], ['m_b'], [], ['measurement of test_obs'])
        with self.assertRaises(AssertionError):
            # non-existent fit parameter
            Fit('test_fit_1', par, wc, ['blabla'],  [], [], ['measurement of test_obs'])
        with self.assertRaises(AssertionError):
            # non-existent nuisance parameter
            Fit('test_fit_1', par, wc, [],  ['blabla'], [], ['measurement of test_obs'])
        with self.assertRaises(AssertionError):
            # non-existent Wilson coefficient
            Fit('test_fit_1', par, wc, ['m_b'],  ['m_c'], ['C_xxx'], ['measurement of test_obs'])
        fit = Fit('test_fit_1', par, wc, ['m_b'],  ['m_c'], ['CVLL_bsbs'], ['measurement of test_obs'])
        self.assertEqual(fit.fit_parameters, ['m_b'])
        self.assertEqual(fit.nuisance_parameters, ['m_c'])
        self.assertEqual(fit.fit_coefficients, ['CVLL_bsbs'])
        self.assertEqual(fit.get_central_fit_parameters, [4.2])
        self.assertEqual(fit.get_central_nuisance_parameters, [1.2])
        # removing dummy instances
        Fit.del_instance('test_fit_1')
        Observable.del_instance('test_obs')
        Measurement.del_instance('measurement of test_obs')

    def test_bayesian_fit_class(self):
        o = Observable( 'test_obs' )
        def f(wc_obj, par_dict):
            return par_dict['m_b']*2
        pr  = Prediction( 'test_obs', f )
        d = NormalDistribution(4.2, 0.2)
        m = Measurement( 'measurement of test_obs' )
        m.add_constraint(['test_obs'], d)
        par = copy.deepcopy(flavio.parameters.default_parameters)
        par.set_constraint('m_b', '4.2+-0.2')
        par.set_constraint('m_c', '1.2+-0.1')
        par.set_constraint('m_s', '0.10(5)')
        wc = flavio.physics.eft.WilsonCoefficients()
        fit = BayesianFit('bayesian_test_fit_1', par, wc, ['m_b','m_c'],  ['m_s'], ['CVLL_bsbs','CVRR_bsbs'], ['measurement of test_obs'])
        self.assertEqual(fit.dimension, 7)
        # test from array to dict ...
        d = fit.array_to_dict(np.array([1.,2.,3.,4.,5.,6.,7.]))
        self.assertEqual(d, {'nuisance_parameters': {'m_s': 3.0}, 'fit_parameters': {'m_c': 2.0, 'm_b': 1.0}, 'fit_coefficients': {'CVLL_bsbs': 4.0+5.j, 'CVRR_bsbs': 6.0+7.0j}})
        # ... and back
        np.testing.assert_array_equal(fit.dict_to_array(d), np.array([1.,2.,3.,4.,5.,6.,7.]))
        self.assertEqual(fit.get_random.shape, (7,))
        fit.log_prior_parameters(np.array([4.5,1.0,0.08,4.,5.,6.,7.]))
        fit.get_predictions(np.array([4.5,1.0,0.08,4.,5.,6.,7.]))
        fit.log_likelihood(np.array([4.5,1.0,0.08,4.,5.,6.,7.]))
        # removing dummy instances
        BayesianFit.del_instance('bayesian_test_fit_1')
        Observable.del_instance('test_obs')
        Measurement.del_instance('measurement of test_obs')
