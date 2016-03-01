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
        p.set_description('b quark mass')
        self.assertEqual( p.description, 'b quark mass' )
        # removing dummy instances
        Parameter.del_instance('test_mb')

    def test_constraints_class(self):
        p = Parameter( 'test_mb' )
        self.assertEqual( p, Parameter.get_instance('test_mb') )
        c = ParameterConstraints()
        d = NormalDistribution(4.2, 0.2)
        c.add_constraint( ['test_mb'], d )
        # checking central values
        self.assertEqual( c.get_central('test_mb'), 4.2)
        # checking types and shapes of random values
        self.assertEqual( type(d.get_random()), float)
        self.assertEqual( d.get_random(3).shape, (3,))
        self.assertEqual( d.get_random((4,5)).shape, (4,5))
        pc = Parameter( 'test_mc' )
        c.add_constraint( ['test_mc', 'test_mb'], MultivariateNormalDistribution([1.2,4.2],[[0.01,0],[0,0.04]]) )
        c.get_logprobability_all(c.get_central_all())
        # removing dummy instances
        Parameter.del_instance('test_mb')
        Parameter.del_instance('test_mc')

    def test_observable_class(self):
        o = Observable( 'test_obs' )
        self.assertEqual( o, Observable.get_instance('test_obs') )
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
