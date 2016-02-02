import unittest
import numpy as np
from .classes import *

class TestClasses(unittest.TestCase):
    def test_parameter_class(self):
        p = Parameter( 'm_b' )
        self.assertEqual( p, Parameter.get_instance('m_b') )
        p.set_description('b quark mass')
        self.assertEqual( p.get_description(), 'b quark mass' )
        d = NormalDistibution(4.2, 0.2)
        p.add_constraint( d )
        # checking central values
        self.assertEqual( p.get_central(), 4.2)
        self.assertEqual( d.get_central(), 4.2)
        # checking types and shapes of random values
        self.assertEqual( type(d.get_random()), float)
        self.assertEqual( d.get_random(3).shape, (3,))
        self.assertEqual( d.get_random((4,5)).shape, (4,5))
        self.assertEqual( type(p.get_random()), np.float64)
        self.assertEqual( p.get_random(3).shape, (3,))
        self.assertEqual( p.get_random((4,5)).shape, (4,5))
        p_c = Parameter( 'm_c' )
        p_c.add_constraint( NormalDistibution(1.2, 0.1) )
        self.assertEqual( Parameter.get_central_all(), {'m_b': 4.2, 'm_c': 1.2} )
        Parameter.get_random_all()

    def test_observable_class(self):
        o = Observable( 'test_obs' )
        self.assertEqual( o, Observable.get_instance('test_obs') )
        o.set_description('some test observables')
        self.assertEqual( o.get_description(), 'some test observables' )
