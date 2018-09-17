import unittest
import numpy as np
import numpy.testing as npt
import flavio
from .fits import *
from flavio.classes import *
from flavio.statistics.probability import *
from flavio.config import config
import scipy.stats
import copy
import os
import tempfile

def fit_wc_fct_error(X):
    raise ValueError("Oops ... this should not have happened")

class TestClasses(unittest.TestCase):
    def test_fit_class(self):
        o = Observable( 'test_obs' )
        d = NormalDistribution(4.2, 0.2)
        par = copy.deepcopy(flavio.parameters.default_parameters)
        par.set_constraint('m_b', '4.2+-0.2')
        par.set_constraint('m_c', '1.2+-0.1')
        with self.assertRaises(AssertionError):
            # unconstrained observable
            Fit('test_fit_1', par, ['m_b'],  ['m_c'], ['test_obs'])
        m = Measurement( 'measurement of test_obs' )
        m.add_constraint(['test_obs'], d)
        with self.assertRaises(AssertionError):
            # same parameter as fit and nuisance
            Fit('test_fit_1', par, ['m_b'], ['m_b'], ['test_obs'])
        with self.assertRaises(AssertionError):
            # non-existent nuisance parameter
            Fit('test_fit_1', par, [],  ['blabla'], ['test_obs'])
        def wc_fct(C):
            return {'CVLL_bsbs': C}
        with self.assertRaises(ValueError):
            # specify include_measurements and exclude_measurements simultaneously
            Fit('test_fit_1', par, ['m_b'],  ['m_c'], ['test_obs'], fit_wc_function=wc_fct,
            include_measurements=['measurement of test_obs'],
            exclude_measurements=['measurement of test_obs'])
        fit = Fit('test_fit_1', par, ['m_b'],  ['m_c'], ['test_obs'], fit_wc_function=wc_fct)
        self.assertEqual(fit.fit_parameters, ['m_b'])
        self.assertEqual(fit.nuisance_parameters, ['m_c'])
        self.assertEqual(fit.fit_wc_names, ('C',))
        self.assertEqual(fit.get_measurements, ['measurement of test_obs'])
        self.assertEqual(fit.get_central_fit_parameters, [4.2])
        self.assertEqual(fit.get_central_nuisance_parameters, [1.2])
        # removing dummy instances
        Fit.del_instance('test_fit_1')
        Observable.del_instance('test_obs')
        Measurement.del_instance('measurement of test_obs')

    def test_correlation_warning(self):
        o1 = Observable( 'test_obs 1' )
        o2 = Observable( 'test_obs 2' )
        d1 = MultivariateNormalDistribution([1,2],[[1,0],[0,2]])
        d2 = MultivariateNormalDistribution([1,2],[[1,0],[0,2]])
        par = flavio.default_parameters
        m1 = Measurement( '1st measurement of test_obs 1 and 2' )
        m1.add_constraint(['test_obs 1', 'test_obs 2'], d1)
        # this should not prompt a warning
        Fit('test_fit_1', par, [], [], observables=['test_obs 1'])
        m2 = Measurement( '2nd measurement of test_obs 1 and 2' )
        m2.add_constraint(['test_obs 1', 'test_obs 2'], d2)
        # this should now prompt a warning
        with self.assertWarnsRegex(UserWarning,
                                   ".*test_fit_1.*test_obs 2.*test_obs 1.*"):
            Fit('test_fit_1', par, [], [], observables=['test_obs 1'])
        Fit.del_instance('test_fit_1')
        Observable.del_instance('test_obs 1')
        Observable.del_instance('test_obs 2')
        Measurement.del_instance('1st measurement of test_obs 1 and 2')
        Measurement.del_instance('2nd measurement of test_obs 1 and 2')

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
        fit = BayesianFit('bayesian_test_fit_1', par, ['m_b','m_c'],  ['m_s'], [('test_obs 2',3)], fit_wc_function=wc_fct)
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
        fit.log_likelihood_exp(np.array([4.5,1.0,0.08,4.,5.]))
        # removing dummy instances
        BayesianFit.del_instance('bayesian_test_fit_1')
        Observable.del_instance('test_obs 2')
        Measurement.del_instance('measurement of test_obs 2')

    def test_fastfit(self):
        # dummy observables
        o1 = Observable( 'test_obs 1' )
        o2 = Observable( 'test_obs 2' )
        # dummy predictions
        def f1(wc_obj, par_dict):
            return par_dict['m_b']
        def f2(wc_obj, par_dict):
            return 2.5
        Prediction( 'test_obs 1', f1 )
        Prediction( 'test_obs 2', f2 )
        d1 = NormalDistribution(5, 0.2)
        cov2 = [[0.1**2, 0.5*0.1*0.3], [0.5*0.1*0.3, 0.3**2]]
        d2 = MultivariateNormalDistribution([6,2], cov2)
        m1 = Measurement( 'measurement 1 of test_obs 1' )
        m2 = Measurement( 'measurement 2 of test_obs 1 and test_obs 2' )
        m1.add_constraint(['test_obs 1'], d1)
        m2.add_constraint(['test_obs 1', 'test_obs 2'], d2)
        fit2 = FastFit('fastfit_test_2', flavio.default_parameters, ['m_b'],  [], ['test_obs 1', 'test_obs 2'])
        # fit with only a single observable and measurement
        fit1 = FastFit('fastfit_test_1', flavio.default_parameters, ['m_b'],  [], ['test_obs 2',])
        for fit in (fit2, fit1):
            fit.make_measurement()
            centr_cov_exp_before = fit._exp_central_covariance
            filename = os.path.join(tempfile.gettempdir(), 'tmp.p')
            fit.save_exp_central_covariance(filename)
            fit.load_exp_central_covariance(filename)
            centr_cov_exp_after = fit._exp_central_covariance
            npt.assert_array_equal(centr_cov_exp_before[0], centr_cov_exp_after[0])
            npt.assert_array_equal(centr_cov_exp_before[1], centr_cov_exp_after[1])
            os.remove(filename)
            cov_before = fit._sm_covariance
            filename = os.path.join(tempfile.gettempdir(), 'tmp-no-p')
            fit.save_sm_covariance(filename)
            fit.load_sm_covariance(filename)
            cov_after = fit._sm_covariance
            npt.assert_array_equal(cov_before, cov_after)
            os.remove(filename)
            filename = os.path.join(tempfile.gettempdir(), 'tmp.p')
            fit.save_sm_covariance(filename)
            fit.load_sm_covariance(filename)
            cov_after = fit._sm_covariance
            npt.assert_array_equal(cov_before, cov_after)
            os.remove(filename)
        fit = fit2  # the following is only for fit2
        cov_weighted = [[0.008, 0.012],[0.012,0.0855]]
        mean_weighted = [5.8, 1.7]
        exact_log_likelihood = scipy.stats.multivariate_normal.logpdf([5.9, 2.5], mean_weighted, cov_weighted)
        self.assertAlmostEqual(fit.log_likelihood([5.9]), exact_log_likelihood, delta=0.8)
        self.assertAlmostEqual(fit.best_fit()['x'], 5.9, delta=0.1)
        # removing dummy instances
        FastFit.del_instance('fastfit_test_1')
        FastFit.del_instance('fastfit_test_2')
        Observable.del_instance('test_obs 1')
        Observable.del_instance('test_obs 2')
        Measurement.del_instance('measurement 1 of test_obs 1')
        Measurement.del_instance('measurement 2 of test_obs 1 and test_obs 2')


    def test_fastfit_covariance_sm(self):
        # This test is to assure that calling make_measurement does not
        # actually call the fit_wc_function
        # dummy observables
        o1 = Observable( 'test_obs 1' )
        # dummy predictions
        def f1(wc_obj, par_dict):
            return par_dict['m_b']
        Prediction( 'test_obs 1', f1 )
        d1 = NormalDistribution(5, 0.2)
        m1 = Measurement( 'measurement 1 of test_obs 1' )
        m1.add_constraint(['test_obs 1'], d1)
        def fit_wc_fct_tmp(X):
            pass
        fit = FastFit('fastfit_test_1', flavio.default_parameters, ['m_b'],  [], ['test_obs 1'],
                      fit_wc_function=fit_wc_fct_tmp)
        fit.fit_wc_function = fit_wc_fct_error
        fit.fit_wc_names = tuple(inspect.signature(fit.fit_wc_function).parameters.keys())
        fit.make_measurement() # single thread calculation
        FastFit.del_instance('fastfit_test_1')
        fit = FastFit('fastfit_test_1', flavio.default_parameters, ['m_b'],  [], ['test_obs 1'],
                      fit_wc_function=fit_wc_fct_tmp)
        fit.fit_wc_function = fit_wc_fct_error
        fit.fit_wc_names = tuple(inspect.signature(fit.fit_wc_function).parameters.keys())
        fit.make_measurement(threads=2) # multi thread calculation
        FastFit.del_instance('fastfit_test_1')
        Observable.del_instance('test_obs 1')
        Measurement.del_instance('measurement 1 of test_obs 1')

    def test_frequentist_fit_class(self):
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
        fit = FrequentistFit('frequentist_test_fit_1', par, ['m_b','m_c'],  ['m_s'], [('test_obs 2',3)], fit_wc_function=wc_fct)
        self.assertEqual(fit.get_measurements, ['measurement of test_obs 2'])
        self.assertEqual(fit.dimension, 5)
        # test from array to dict ...
        d = fit.array_to_dict(np.array([1.,2.,3.,4.,5.]))
        self.assertEqual(d, {'nuisance_parameters': {'m_s': 3.0}, 'fit_parameters': {'m_c': 2.0, 'm_b': 1.0}, 'fit_wc': {'CL': 4.0, 'CR': 5.0}})
        # ... and back
        np.testing.assert_array_equal(fit.dict_to_array(d), np.array([1.,2.,3.,4.,5.]))
        fit.log_prior_parameters(np.array([4.5,1.0,0.08,4.,5.]))
        fit.get_predictions(np.array([4.5,1.0,0.08,4.,5.]))
        fit.log_likelihood_exp(np.array([4.5,1.0,0.08,4.,5.]))
        fit.log_likelihood(np.array([4.5,1.0,0.08,4.,5.]))
        # removing dummy instances
        FrequentistFit.del_instance('frequentist_test_fit_1')
        Observable.del_instance('test_obs 2')
        Measurement.del_instance('measurement of test_obs 2')
