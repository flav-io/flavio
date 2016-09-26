import unittest
import numpy as np
import flavio
from .fits import *
from flavio.classes import *
from flavio.statistics.probability import *
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
        with self.assertRaises(AssertionError):
            # same parameter as fit and nuisance
            Fit('test_fit_1', par, ['m_b'], ['m_b'], ['test_obs'])
        with self.assertRaises(AssertionError):
            # non-existent fit parameter
            Fit('test_fit_1', par, ['blabla'],  [], ['test_obs'])
        with self.assertRaises(AssertionError):
            # non-existent nuisance parameter
            Fit('test_fit_1', par, [],  ['blabla'], ['test_obs'])
        def wc_fct(C):
            return {'CVLL_bsbs': C}
        with self.assertRaises(ValueError):
            # specify include_measurements and exclude_measurements simultaneously
            Fit('test_fit_1', par, ['m_b'],  ['m_c'], ['test_obs'], ['C'], wc_fct,
            include_measurements=['measurement of test_obs'],
            exclude_measurements=['measurement of test_obs'])
        fit = Fit('test_fit_1', par, ['m_b'],  ['m_c'], ['test_obs'], ['C'], wc_fct)
        self.assertEqual(fit.fit_parameters, ['m_b'])
        self.assertEqual(fit.nuisance_parameters, ['m_c'])
        self.assertEqual(fit.fit_wc_names, ['C'])
        self.assertEqual(fit.get_measurements, ['measurement of test_obs'])
        self.assertEqual(fit.get_central_fit_parameters, [4.2])
        self.assertEqual(fit.get_central_nuisance_parameters, [1.2])
        # removing dummy instances
        Fit.del_instance('test_fit_1')
        Observable.del_instance('test_obs')
        Measurement.del_instance('measurement of test_obs')

    def test_bayesian_fit_class(self):
        o = Observable( 'test_obs 2' )
        o.arguments = ['q2']
        def f(wc_obj, par_dict, q2):
            return par_dict['m_b']*2 *q2
        pr  = Prediction( 'test_obs 2', f )
        d = NormalDistribution(4.2, 0.2)
        m = Measurement( 'measurement of test_obs 2' )
        m.add_constraint([('test_obs 2', 3)], d)
        par = copy.deepcopy(flavio.parameters.default_parameters)
        par.set_constraint('m_b', '4.2+-0.2')
        par.set_constraint('m_c', '1.2+-0.1')
        par.set_constraint('m_s', '0.10(5)')
        wc = flavio.physics.eft.WilsonCoefficients()
        def wc_fct(CL, CR):
            return {'CVLL_bsbs': CL, 'CVRR_bsbs': CR}
        fit = BayesianFit('bayesian_test_fit_1', par, ['m_b','m_c'],  ['m_s'], [('test_obs 2',3)], ['CL','CR'], wc_fct)
        self.assertEqual(fit.get_measurements, ['measurement of test_obs 2'])
        self.assertEqual(fit.dimension, 5)
        # test from array to dict ...
        d = fit.array_to_dict(np.array([1.,2.,3.,4.,5.]))
        self.assertEqual(d, {'nuisance_parameters': {'m_s': 3.0}, 'fit_parameters': {'m_c': 2.0, 'm_b': 1.0}, 'fit_wc': {'CL': 4.0, 'CR': 5.0}})
        # ... and back
        np.testing.assert_array_equal(fit.dict_to_array(d), np.array([1.,2.,3.,4.,5.]))
        self.assertEqual(fit.get_random.shape, (5,))
        fit.log_prior_parameters(np.array([4.5,1.0,0.08,4.,5.]))
        fit.get_predictions(np.array([4.5,1.0,0.08,4.,5.]))
        fit.log_likelihood(np.array([4.5,1.0,0.08,4.,5.]))
        # removing dummy instances
        BayesianFit.del_instance('bayesian_test_fit_1')
        Observable.del_instance('test_obs 2')
        Measurement.del_instance('measurement of test_obs 2')
