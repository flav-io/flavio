import unittest
import numpy as np
from flavio.classes import *
from flavio.statistics.probability import *
from flavio.config import config
import scipy.integrate
import math

class TestClasses(unittest.TestCase):
    def test_parameter_class(self):
        p = Parameter( 'test_mb' )
        self.assertEqual( p, Parameter.get_instance('test_mb') )
        self.assertEqual( p, Parameter['test_mb'] )
        p.set_description('b quark mass')
        self.assertEqual( p.description, 'b quark mass' )
        # removing dummy instances
        Parameter.del_instance('test_mb')

    def test_constraints_class(self):
        p = Parameter( 'test_mb' )
        self.assertEqual( p, Parameter.get_instance('test_mb') )
        self.assertEqual( p, Parameter['test_mb'] )
        c = ParameterConstraints()
        d = NormalDistribution(4.2, 0.2)
        c.add_constraint( ['test_mb'], d )
        # checking central values
        self.assertEqual( c.get_central('test_mb'), 4.2)
        # checking types and shapes of random values
        self.assertEqual( type(d.get_random()), float)
        self.assertEqual( d.get_random(3).shape, (3,))
        self.assertEqual( d.get_random((4,5)).shape, (4,5))
        test_1derrors_random = c.get_1d_errors_random()
        self.assertAlmostEqual( test_1derrors_random['test_mb'], 0.2, delta=0.5)
        test_1derrors_rightleft = c.get_1d_errors_rightleft()
        self.assertEqual( test_1derrors_rightleft['test_mb'], (0.2, 0.2))
        d = AsymmetricNormalDistribution(4.2, 0.2, 0.1)
        c.add_constraint( ['test_mb'], d )
        test_1derrors_rightleft = c.get_1d_errors_rightleft()
        self.assertEqual( test_1derrors_rightleft['test_mb'], (0.2, 0.1))
        pc = Parameter( 'test_mc' )
        c.add_constraint( ['test_mc', 'test_mb'], MultivariateNormalDistribution([1.2,4.2],[[0.01,0],[0,0.04]]) )
        c.get_logprobability_all(c.get_central_all())
        test_1derrors_random = c.get_1d_errors_random()
        self.assertAlmostEqual( test_1derrors_random['test_mb'], 0.2, delta=0.05)
        self.assertAlmostEqual( test_1derrors_random['test_mc'], 0.1, delta=0.05)
        test_1derrors_rightleft = c.get_1d_errors_rightleft()
        self.assertEqual( test_1derrors_rightleft['test_mb'], (0.2, 0.2))
        self.assertEqual( test_1derrors_rightleft['test_mc'], (0.1, 0.1))
        # removing dummy instances
        # check that they have been removed, using old and new syntax
        c.remove_constraint('test_mb')
        Parameter.del_instance('test_mb')
        with self.assertRaises(KeyError):
            Parameter['test_mb']
        del Parameter['test_mc']
        with self.assertRaises(KeyError):
            Parameter.get_instance('test_mc')

    def test_set_constraint(self):
        p = Parameter( 'test_mb' )
        c = ParameterConstraints()
        c.set_constraint('test_mb', '4.2 +- 0.1 +- 0.2')
        cons = c._parameters['test_mb'][1]
        self.assertIsInstance(cons, NormalDistribution)
        self.assertEqual(cons.central_value, 4.2)
        self.assertEqual(cons.standard_deviation, math.sqrt(0.1**2+0.2**2))
        c.set_constraint('test_mb', '4.3 + 0.3 - 0.4')
        cons = c._parameters['test_mb'][1]
        self.assertIsInstance(cons, AsymmetricNormalDistribution)
        self.assertEqual(cons.central_value, 4.3)
        self.assertEqual(cons.right_deviation, 0.3)
        self.assertEqual(cons.left_deviation, 0.4)
        c.set_constraint('test_mb', 4.4)
        cons = c._parameters['test_mb'][1]
        self.assertIsInstance(cons, DeltaDistribution)
        self.assertEqual(cons.central_value, 4.4)
        cons_dict_1 = {'distribution': 'normal',
                     'central_value': '4.5',
                     'standard_deviation': '0.1'}
        cons_dict_2 = {'distribution': 'normal',
                     'central_value': 4.5,
                     'standard_deviation': 0.2}
        c.set_constraint('test_mb', constraint_dict=cons_dict_1)
        cons = c._parameters['test_mb'][1]
        self.assertIsInstance(cons, NormalDistribution)
        self.assertEqual(cons.central_value, 4.5)
        self.assertEqual(cons.standard_deviation, 0.1)
        c.set_constraint('test_mb', constraint_dict=[cons_dict_1, cons_dict_2])
        cons = c._parameters['test_mb'][1]
        self.assertIsInstance(cons, NormalDistribution)
        self.assertEqual(cons.central_value, 4.5)
        self.assertEqual(cons.standard_deviation, math.sqrt(0.1**2+0.2**2))
        Parameter.del_instance('test_mb')

    def test_observable_class(self):
        o = Observable( 'test_obs' )
        self.assertEqual( o, Observable.get_instance('test_obs') )
        self.assertEqual( o, Observable['test_obs'] )
        o.set_description('some test observables')
        self.assertEqual( o.description, 'some test observables' )
        # removing dummy instances
        Observable.del_instance('test_obs')

    def test_measurement_class(self):
        o = Observable( 'test_obs' )
        d = NormalDistribution(4.2, 0.2)
        m = Measurement( 'measurement of test_obs' )
        m.add_constraint(['test_obs'], d)
        # removing dummy instances
        Observable.del_instance('test_obs')
        Measurement.del_instance('measurement of test_obs')

    def test_prediction_class(self):
        o = Observable( 'test_obs' )
        p = Parameter( 'test_parameter' )
        def f(wc_obj, par_dict):
            return par_dict['test_parameter']*2
        pr  = Prediction( 'test_obs', f )
        wc_obj = None
        c = ParameterConstraints()
        c.add_constraint( ['test_parameter'], NormalDistribution(1.2, 0.1) )
        self.assertEqual( pr.get_central(c, wc_obj), 2.4)
        self.assertEqual( o.prediction_central(c, wc_obj), 2.4)
        # removing dummy instances
        Observable.del_instance('test_obs')
        Parameter.del_instance('test_parameter')

    def test_implementation_class(self):
        a = AuxiliaryQuantity( 'test_aux' )
        p = Parameter( 'test_parameter' )
        def f(wc_obj, par_dict):
            return par_dict['test_parameter']*2
        imp  = Implementation( 'test_imp', 'test_aux', f )
        config['implementation']['test_aux'] = 'test_imp'
        wc_obj = None
        c = ParameterConstraints()
        c.add_constraint( ['test_parameter'], NormalDistribution(1.2, 0.1) )
        self.assertEqual( imp.get_central(c, wc_obj), 2.4)
        self.assertEqual( a.prediction_central(c, wc_obj), 2.4)
        Implementation.show_all()
        # removing dummy instances
        AuxiliaryQuantity.del_instance('test_aux')
        Parameter.del_instance('test_parameter')
        Implementation.del_instance('test_imp')
        del config['implementation']['test_aux']

    def test_constraints_class_exclude(self):
        Parameter( 'test_ma' )
        Parameter( 'test_mb' )
        Parameter( 'test_mc' )
        c2 = np.array([1e-3, 2])
        c3 = np.array([1e-3, 2, 0.4])
        cov22 = np.array([[(0.2e-3)**2, 0.2e-3*0.5*0.3],[0.2e-3*0.5*0.3, 0.5**2]])
        cov33 = np.array([[(0.2e-3)**2, 0.2e-3*0.5*0.3 , 0],[0.2e-3*0.5*0.3, 0.5**2, 0.01], [0, 0.01, 0.1**2]])
        pdf1 = NormalDistribution(2, 0.5)
        pdf2 = MultivariateNormalDistribution(c2, cov22)
        pdf3 = MultivariateNormalDistribution(c3, cov33)
        c1 = ParameterConstraints()
        c2 = ParameterConstraints()
        c3 = ParameterConstraints()
        c1.add_constraint(['test_mb'], pdf1)
        c2.add_constraint(['test_ma', 'test_mb'], pdf2)
        c3.add_constraint(['test_ma', 'test_mb', 'test_mc'], pdf3)
        par_dict = {'test_ma': 1.2e-3, 'test_mb': 2.4, 'test_mc': 0.33}
        self.assertEqual(
            c1.get_logprobability_all(par_dict)[pdf1],
            c2.get_logprobability_all(par_dict, exclude_parameters=['test_ma', 'test_mc'])[pdf2],
        )
        self.assertEqual(
            c1.get_logprobability_all(par_dict)[pdf1],
            c3.get_logprobability_all(par_dict, exclude_parameters=['test_ma', 'test_mc'])[pdf3],
        )
        self.assertEqual(
            c2.get_logprobability_all(par_dict)[pdf2],
            c3.get_logprobability_all(par_dict, exclude_parameters=['test_mc'])[pdf3],
        )
        # remove dummy instances
        Parameter.del_instance('test_ma')
        Parameter.del_instance('test_mb')
        Parameter.del_instance('test_mc')


    def test_pdf(self):
        # for the normal dist's, just check that no error is raised
        pd = NormalDistribution(1., 0.2)
        pd.logpdf(0.5)
        pd = MultivariateNormalDistribution([1.,2.], [[0.04,0],[0,0.09]])
        pd.logpdf([1.05,2.08])
        # for the asymmetric dist, more scrutiny needed
        pd = AsymmetricNormalDistribution(1., 0.2, 0.5)
        eps = 1.e-8
        # check that the PDF is continuos
        self.assertAlmostEqual( pd.logpdf(1. - eps), pd.logpdf(1. + eps), places=8)
        # check that the PDF is properly normalized
        self.assertEqual( scipy.integrate.quad(lambda x: math.exp(pd.logpdf(x)), -np.inf, +np.inf)[0], 1)

    def test_observable_taxonomy(self):
        o1 = Observable( 'test_obs_1' )
        o2 = Observable( 'test_obs_2' )
        o1.add_taxonomy('test 1 :: test 2 :: test 3')
        o2.add_taxonomy('test 1 :: test 2 :: test 3')
        self.assertDictEqual(
            Observable.taxonomy_dict()['test 1'],
            {'test 2': {'test 3': {'test_obs_1' :{}, 'test_obs_2':{}}}}
        )
        # remove test from taxonomy
        Observable.taxonomy.pop('test 1', None)
        # removing dummy instances
        Observable.del_instance('test_obs_1')
        Observable.del_instance('test_obs_2')
