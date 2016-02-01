import unittest
import numpy as np
from .classes import *

class TestClasses(unittest.TestCase):
    def test_parameter_class(self):
        p = Parameter( ('mass', 'b') )
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
