import unittest
import flavio
from .likelihood import *
from flavio.classes import *
from flavio.statistics.probability import *
import numpy.testing as npt
import voluptuous as vol
import os
import tempfile


class TestMeasurementLikelihood(unittest.TestCase):
    def test_class(self):
        o = Observable( 'test_obs' )
        def f(wc_obj, par_dict):
            return par_dict['m_b']*2
        Prediction('test_obs', f)
        d = NormalDistribution(4.2, 0.2)
        m = Measurement( 'measurement of test_obs' )
        m.add_constraint(['test_obs'], d)
        with self.assertRaises(ValueError):
            # specify include_measurements and exclude_measurements simultaneously
            MeasurementLikelihood(['test_obs'],
                include_measurements=['measurement of test_obs'],
                exclude_measurements=['measurement of test_obs'])
        ml = MeasurementLikelihood(['test_obs'])
        pred = ml.get_predictions_par({'m_b': 4}, None)
        self.assertDictEqual(pred, {'test_obs': 8})
        self.assertEqual(ml.get_measurements, ['measurement of test_obs'])
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
        MeasurementLikelihood(observables=['test_obs 1'])
        m2 = Measurement( '2nd measurement of test_obs 1 and 2' )
        m2.add_constraint(['test_obs 1', 'test_obs 2'], d2)
        # this should now prompt a warning
        with self.assertWarnsRegex(UserWarning,
                                   ".*test_obs 2.*test_obs 1.*"):
            MeasurementLikelihood(observables=['test_obs 1'])
        Observable.del_instance('test_obs 1')
        Observable.del_instance('test_obs 2')
        Measurement.del_instance('1st measurement of test_obs 1 and 2')
        Measurement.del_instance('2nd measurement of test_obs 1 and 2')

    def test_load(self):
        d = {}
        o = Observable( 'test_obs' )
        def f(wc_obj, par_dict):
            return par_dict['m_b']*2
        Prediction('test_obs', f)
        d = NormalDistribution(4.2, 0.2)
        m = Measurement( 'measurement of test_obs' )
        m.add_constraint(['test_obs'], d)
        with self.assertRaises(vol.error.Error):
            # string instead of list
            ml = MeasurementLikelihood.load_dict({'observables' : 'test_obs'})
        with self.assertRaises(TypeError):
            # compulsory argument missing
            ml = MeasurementLikelihood.load_dict({})
        # should work
        ml = MeasurementLikelihood.load_dict({'observables' : ['test_obs']})
        pred = ml.get_predictions_par({'m_b': 4}, None)
        self.assertDictEqual(pred, {'test_obs': 8})
        self.assertEqual(ml.get_measurements, ['measurement of test_obs'])
        self.assertEqual(ml.get_number_observations(), 1)
        m = Measurement( 'measurement 2 of test_obs' )
        m.add_constraint(['test_obs'], d)
        self.assertEqual(ml.get_number_observations(), 2)
        Observable.del_instance('test_obs')
        Measurement.del_instance('measurement of test_obs')
        Measurement.del_instance('measurement 2 of test_obs')


class TestParameterLikelihood(unittest.TestCase):
    def test_parameter_likelihood(self):
        par = copy.deepcopy(flavio.parameters.default_parameters)
        par.set_constraint('m_b', '4.2+-0.2')
        par.set_constraint('m_c', '1.2+-0.1')
        pl = ParameterLikelihood(par, ['m_b', 'm_c'])
        self.assertListEqual(pl.parameters, ['m_b', 'm_c'])
        npt.assert_array_equal(pl.get_central, [4.2, 1.2])
        self.assertEqual(len(pl.get_random), 2)
        # test likelihood
        chi2_central = -2 * pl.log_likelihood_par({'m_b': 4.2, 'm_c': 1.2})
        chi2_2s = -2 * pl.log_likelihood_par({'m_b': 4.6, 'm_c': 1.0})
        self.assertAlmostEqual(chi2_2s - chi2_central, 4 + 4)


    def test_load(self):
        par = copy.deepcopy(flavio.parameters.default_parameters)
        par.set_constraint('m_b', '4.2+-0.2')
        par.set_constraint('m_c', '1.2+-0.1')
        pl = ParameterLikelihood.load_dict({'par_obj': [{'m_b': '4.2+-0.2'},
                                                   {'m_c': '1.2+-0.1'}],
                                       'parameters': ['m_b', 'm_c']})
        self.assertListEqual(pl.parameters, ['m_b', 'm_c'])
        npt.assert_array_equal(pl.get_central, [4.2, 1.2])
        self.assertEqual(len(pl.get_random), 2)
        # test likelihood
        chi2_central = -2 * pl.log_likelihood_par({'m_b': 4.2, 'm_c': 1.2})
        chi2_2s = -2 * pl.log_likelihood_par({'m_b': 4.6, 'm_c': 1.0})
        self.assertAlmostEqual(chi2_2s - chi2_central, 4 + 4)


class TestLikelihood(unittest.TestCase):
    def test_likelihood(self):
        o = Observable( 'test_obs' )
        def f(wc_obj, par_dict):
            return par_dict['m_b']*2
        Prediction('test_obs', f)
        d = NormalDistribution(4.2, 0.2)
        m = Measurement( 'measurement of test_obs' )
        m.add_constraint(['test_obs'], d)
        par = copy.deepcopy(flavio.parameters.default_parameters)
        par.set_constraint('m_b', '4.2+-0.2')
        par.set_constraint('m_c', '1.2+-0.1')
        pl = Likelihood(par, ['m_b', 'm_c'], ['test_obs'])
        # npt.assert_array_equal(pl.get_central, [4.2, 1.2])
        # self.assertEqual(len(pl.get_random), 2)
        # test likelihoods
        chi2_central = -2 * pl.log_prior_fit_parameters({'m_b': 4.2, 'm_c': 1.2})
        chi2_2s = -2 * pl.log_prior_fit_parameters({'m_b': 4.6, 'm_c': 1.2})
        self.assertAlmostEqual(chi2_2s - chi2_central, 4)
        chi2_central = -2 * pl.log_prior_fit_parameters({'m_b': 4.2, 'm_c': 1.2})
        chi2_2s = -2 * pl.log_prior_fit_parameters({'m_b': 4.2, 'm_c': 1.0})
        self.assertAlmostEqual(chi2_2s - chi2_central, 4)
        Observable.del_instance('test_obs')
        Measurement.del_instance('measurement of test_obs')

    def test_load(self):
        o = Observable( 'test_obs' )
        def f(wc_obj, par_dict):
            return par_dict['m_b']*2
        Prediction('test_obs', f)
        d = NormalDistribution(4.2, 0.2)
        m = Measurement( 'measurement of test_obs' )
        m.add_constraint(['test_obs'], d)
        par = copy.deepcopy(flavio.parameters.default_parameters)
        par.set_constraint('m_b', '4.2+-0.2')
        par.set_constraint('m_c', '1.2+-0.1')
        pl = Likelihood.load_dict({'par_obj': [{'m_b': '4.2+-0.2'},
                                               {'m_c': '1.2+-0.1'}],
                                   'fit_parameters': ['m_b', 'm_c'],
                                   'observables': ['test_obs']})
        pl = Likelihood(par, ['m_b', 'm_c'], ['test_obs'])
        # npt.assert_array_equal(pl.get_central, [4.2, 1.2])
        # self.assertEqual(len(pl.get_random), 2)
        # test likelihoods
        chi2_central = -2 * pl.log_prior_fit_parameters({'m_b': 4.2, 'm_c': 1.2})
        chi2_2s = -2 * pl.log_prior_fit_parameters({'m_b': 4.6, 'm_c': 1.2})
        self.assertAlmostEqual(chi2_2s - chi2_central, 4)
        chi2_central = -2 * pl.log_prior_fit_parameters({'m_b': 4.2, 'm_c': 1.2})
        chi2_2s = -2 * pl.log_prior_fit_parameters({'m_b': 4.2, 'm_c': 1.0})
        self.assertAlmostEqual(chi2_2s - chi2_central, 4)
        Observable.del_instance('test_obs')
        Measurement.del_instance('measurement of test_obs')


class TestCovariances(unittest.TestCase):
    def test_sm_covariance(self):
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
        fit2 = SMCovariance(['test_obs 1', 'test_obs 2'], vary_parameters=['m_b'])
        # single observable
        fit1 = SMCovariance(['test_obs 1'], vary_parameters=['m_b'])
        for fit in (fit2, fit1):
            fit.get()
            cov_before = fit._cov
            filename = os.path.join(tempfile.gettempdir(), 'tmp-no-p')
            fit.save(filename)
            fit.load(filename)
            cov_after = fit._cov
            npt.assert_array_equal(cov_before, cov_after)
            os.remove(filename)
            filename = os.path.join(tempfile.gettempdir(), 'tmp.p')
            fit.save(filename)
            fit.load(filename)
            cov_after = fit._cov
            npt.assert_array_equal(cov_before, cov_after)
            os.remove(filename)
        # removing dummy instances
        Observable.del_instance('test_obs 1')
        Observable.del_instance('test_obs 2')

    def test_exp_covariance(self):
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
        fit2 = MeasurementCovariance(MeasurementLikelihood(['test_obs 1', 'test_obs 2']))
        # single observable
        fit1 = MeasurementCovariance(MeasurementLikelihood(['test_obs 1']))
        for fit in (fit2, fit1):
            fit.get()
            cov_before = fit._central_cov[1]
            filename = os.path.join(tempfile.gettempdir(), 'tmp-no-p')
            fit.save(filename)
            fit.load(filename)
            cov_after = fit._central_cov[1]
            npt.assert_array_equal(cov_before, cov_after)
            os.remove(filename)
            filename = os.path.join(tempfile.gettempdir(), 'tmp.p')
            fit.save(filename)
            fit.load(filename)
            cov_after = fit._central_cov[1]
            npt.assert_array_equal(cov_before, cov_after)
            os.remove(filename)
        # removing dummy instances
        Observable.del_instance('test_obs 1')
        Observable.del_instance('test_obs 2')
        Measurement.del_instance('measurement 1 of test_obs 1')
        Measurement.del_instance('measurement 2 of test_obs 1 and test_obs 2')


class TestFastLikelihood(unittest.TestCase):

    def test_fastlikelihood(self):
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
        fit2 = FastLikelihood('fastlh_test_2', flavio.default_parameters, [], ['m_b'], ['test_obs 1', 'test_obs 2'])
        # fit with only a single observable and measurement
        fit1 = FastLikelihood('fastlh_test_1', flavio.default_parameters, [], ['m_b'], ['test_obs 2',])
        fit3 = FastLikelihood('fastlh_test_3', flavio.default_parameters, ['m_b'],  [], ['test_obs 1', 'test_obs 2'],
                       include_measurements=['measurement 2 of test_obs 1 and test_obs 2'])
        for fit in (fit2, fit1):
            fit.make_measurement(N=10000)
        fit = fit2  # the following is only for fit2
        cov_weighted = [[0.008, 0.012], [0.012, 0.0855]]
        mean_weighted = [5.8, 1.7]
        exact_log_likelihood = scipy.stats.multivariate_normal.logpdf([5.9, 2.5], mean_weighted, cov_weighted)
        self.assertAlmostEqual(fit.log_likelihood({'m_b': 5.9}, None), exact_log_likelihood, delta=0.8)
        # self.assertAlmostEqual(fit.best_fit()['x'], 5.9, delta=0.1)
        # removing dummy instances
        FastLikelihood.del_instance('fastlh_test_1')
        FastLikelihood.del_instance('fastlh_test_2')
        Observable.del_instance('test_obs 1')
        Observable.del_instance('test_obs 2')
        Measurement.del_instance('measurement 1 of test_obs 1')
        Measurement.del_instance('measurement 2 of test_obs 1 and test_obs 2')
