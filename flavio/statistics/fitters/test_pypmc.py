import unittest
from flavio.classes import *
from flavio.statistics.fits import *
from .pypmc import *
from flavio.statistics.probability import NormalDistribution, MultivariateNormalDistribution


class TestPypmcScan(unittest.TestCase):

    def test_pypmc_scan(self):
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
        fit = BayesianFit('fit_emcee_test', flavio.default_parameters, ['m_b', 'm_c'],  [], ['test_obs 1', 'test_obs 2'])
        scan = pypmcScan(fit)
        scan.run(3, burnin=0)
        self.assertTupleEqual(scan.result.shape, (3, 2))
        BayesianFit.del_instance('fit_emcee_test')
        Observable.del_instance('test_obs 1')
        Observable.del_instance('test_obs 2')
        Measurement.del_instance('measurement 1 of test_obs 1')
        Measurement.del_instance('measurement 2 of test_obs 1 and test_obs 2')
