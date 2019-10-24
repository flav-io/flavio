import unittest
import numpy as np
import flavio
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

    def test_logprobability_single(self):
        c_corr = Constraints()
        c_uncorr = Constraints()
        d1 = NormalDistribution(2, 0.3)
        c_corr.add_constraint(['par_1'], d1)
        c_uncorr.add_constraint(['par_1'], d1)
        d23 = MultivariateNormalDistribution([4, 5],
            covariance=[[0.2**2, 0.5*0.2*0.3], [0.5*0.2*0.3, 0.3**2]])
        d2 = NormalDistribution(4, 0.2)
        d3 = NormalDistribution(5, 0.3)
        c_corr.add_constraint(['par_2', 'par_3'], d23)
        d23_uncorr = MultivariateNormalDistribution([4, 5], covariance=[[0.2, 0], [0, 0.3]])
        c_uncorr.add_constraint(['par_2'], d2)
        c_uncorr.add_constraint(['par_3'], d3)
        d = {'par_1': 2.8, 'par_2': 4.9, 'par_3': 4.3}
        # all logprobs for the uncorrelated case
        l_all = c_uncorr.get_logprobability_all(d)
        # the dict should contain the same values as the "single" ones in
        # the correlated case
        for k, v in l_all.items():
            par = dict(c_uncorr._constraints)[k][0]
            self.assertEqual(v,
                             c_corr.get_logprobability_single(par, d[par]),
                             msg="Failed for {}".format(par))

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

    def test_observable_from_function(self):
        o1 = Observable('test_obs_1', arguments=['a1'])
        o2 = Observable('test_obs_2', arguments=['a2'])
        with self.assertRaises(ValueError):
            # non-existent obs
            Observable.from_function('test_obs_12',
                                     ['test_obs_x', 'test_obs_2'],
                                     lambda x, y: x-y)
        with self.assertRaises(AssertionError):
            # depend on different arguments
            Observable.from_function('test_obs_12',
                                     ['test_obs_1', 'test_obs_2'],
                                     lambda x, y: x-y)
        o2 = Observable('test_obs_2', arguments=['a1'])
        with self.assertRaises(AssertionError):
            # obs without prediction
            Observable.from_function('test_obs_12',
                                     ['test_obs_1', 'test_obs_2'],
                                     lambda x, y: x-y)
        Prediction('test_obs_1', lambda wc_obj, par: 3)
        Prediction('test_obs_2', lambda wc_obj, par: 7)
        Observable.from_function('test_obs_12',
                                 ['test_obs_1', 'test_obs_2'],
                                 lambda x, y: x-y)
        self.assertEqual(
            Observable['test_obs_12'].prediction_central(flavio.default_parameters, None),
            -4)
        self.assertEqual(Observable['test_obs_12'].arguments, ['a1'])
        # delete dummy instances
        Observable.del_instance('test_obs_1')
        Observable.del_instance('test_obs_2')
        Observable.del_instance('test_obs_12')

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

    def test_parameter_constraints_yaml(self):
        yaml = flavio.default_parameters.get_yaml()
        pnew = ParameterConstraints.from_yaml(yaml)
        yaml2 = pnew.get_yaml()
        self.assertEqual(yaml, yaml2)

    def test_measurements_yaml(self):
        import json
        for m in Measurement.instances.values():
            if (m.name in ['Belle B->D*lnu hadronic tag 2017',
                           'CLEO D->Kenu 2009',
                           'CLEO D->pienu 2009']
            or 'Pseudo-measurement' in m.name):
                continue  # known failures ...
            yaml = m.get_yaml_dict()
            mnew = Measurement.from_yaml_dict(yaml)
            yaml2 = mnew.get_yaml_dict()
            self.assertEqual(yaml, yaml2, msg="Failed for {}".format(m.name))

    def test_from_yaml_dict(self):
        c = Constraints.from_yaml_dict([
        {'my_par_1': '1 +- 0.3 +- 0.4'},
        {'parameters': ['my_par_2'],
         'values': {
           'distribution': 'normal',
           'central_value': 2,
           'standard_deviation': 0.6,
           }
        },
        {'parameters': ['my_par_3'],
         'values': [{
           'distribution': 'normal',
           'central_value': 7,
           'standard_deviation': 0.3,
           },
           {
           'distribution': 'normal',
           'central_value': 7,
           'standard_deviation': 0.4,
         }]

        }
        ])
        self.assertListEqual(list(c._parameters.keys()), ['my_par_1', 'my_par_2', 'my_par_3'])
        c1 = c._parameters['my_par_1'][1]
        self.assertEqual(type(c1), NormalDistribution)
        self.assertEqual(c1.central_value, 1)
        self.assertEqual(c1.standard_deviation, 0.5)
        c2 = c._parameters['my_par_2'][1]
        self.assertEqual(type(c2), NormalDistribution)
        self.assertEqual(c2.central_value, 2)
        self.assertEqual(c2.standard_deviation, 0.6)
        c3 = c._parameters['my_par_3'][1]
        self.assertEqual(type(c3), NormalDistribution)
        self.assertEqual(c3.central_value, 7)
        self.assertEqual(c3.standard_deviation, 0.5)
        del c

    def test_repr_meas(self):
        mtest = Measurement('repr test')
        self.assertEqual(repr(mtest), "Measurement('repr test')")
        mtest._repr_markdown_()
        mtest.description = "bla"
        self.assertIn("bla", mtest._repr_markdown_())
        mtest.url = "blo"
        self.assertIn("blo", mtest._repr_markdown_())
        del Measurement['repr test']

    def test_repr_obs(self):
        mtest = Observable('repr test')
        self.assertEqual(repr(mtest),
                         "Observable('repr test', arguments=None)")
        mtest._repr_markdown_()
        mtest.description = "bla"
        self.assertIn("bla", mtest._repr_markdown_())
        mtest.tex = "blo"
        self.assertIn("blo", mtest._repr_markdown_())
        mtest.arguments = ["blu"]
        self.assertIn("blu", mtest._repr_markdown_())
        self.assertEqual(repr(mtest),
                         "Observable('repr test', arguments=['blu'])")
        del Observable['repr test']

    def test_repr_par(self):
        ptest = Parameter('repr test')
        self.assertEqual(repr(ptest),
                         "Parameter('repr test')")
        ptest._repr_markdown_()
        ptest.description = "bla"
        self.assertIn("bla", ptest._repr_markdown_())
        ptest.tex = "blo"
        self.assertIn("blo", ptest._repr_markdown_())
        del Parameter['repr test']

    def test_argument_format(self):
        with self.assertRaises(KeyError):
            Observable.argument_format('dont_exist')
        with self.assertRaises(KeyError):
            Observable.argument_format(['dont_exist', 1])
        with self.assertRaises(ValueError):
            Observable.argument_format(['eps_K', 1])
        with self.assertRaises(KeyError):
            Observable.argument_format({'name': 'dBR/dq2(B0->Denu)', 'bla': 1})
        with self.assertRaises(ValueError):
            Observable.argument_format(['dBR/dq2(B0->Denu)'])
        with self.assertRaises(ValueError):
            Observable.argument_format('dBR/dq2(B0->Denu)')
        self.assertTupleEqual(Observable.argument_format({'name': 'dBR/dq2(B0->Denu)', 'q2': 1}, 'tuple'),
                              ('dBR/dq2(B0->Denu)', 1))
        self.assertListEqual(Observable.argument_format({'name': 'dBR/dq2(B0->Denu)', 'q2': 1}, 'list'),
                              ['dBR/dq2(B0->Denu)', 1])
        self.assertDictEqual(Observable.argument_format({'name': 'dBR/dq2(B0->Denu)', 'q2': 1}, 'dict'),
                              {'name': 'dBR/dq2(B0->Denu)', 'q2': 1})
        self.assertEqual(Observable.argument_format('eps_K', 'tuple'),
                              'eps_K')
        self.assertEqual(Observable.argument_format('eps_K', 'list'),
                              'eps_K')
        self.assertDictEqual(Observable.argument_format('eps_K', 'dict'),
                              {'name': 'eps_K'})

    def test_obs_get_meas(self):
        self.assertEqual(
            Observable['eps_K'].get_measurements(),
            ['PDG kaon CPV']
        )
        self.assertEqual(
            sorted(Observable['<BR>(B+->rholnu)'].get_measurements()),
            sorted(['Belle B+->rholnu 2013', 'BaBar B+->rholnu 2010'])
        )
        self.assertEqual(
            Observable['FL(B0->K*mumu)'].get_measurements(),
            []
        )

    def test_find(self):
        class TestClass(NamedInstanceClass):
            pass

        TestClass('test word 1')
        TestClass('x test word 2')
        TestClass('test 3')
        self.assertEqual(TestClass.find('word'), ['test word 1', 'x test word 2'])
        self.assertEqual(TestClass.find('^x'), ['x test word 2'])
        self.assertEqual(TestClass.find('s.*3'), ['test 3'])
