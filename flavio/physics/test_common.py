import unittest
import numpy as np
from .common import *

wc   = {'C1': 1. + 5.j, 'C2': np.array([1,2,3], dtype=float), 'C3': np.array([1.j]) }
wc_c = {'C1': 1. - 5.j, 'C2': np.array([1,2,3], dtype=float), 'C3': np.array([-1.j]) }
par  = {'a': 3., 'b': 4.+5.j, 'delta':  0.2}
par_c= {'a': 3., 'b': 4.+5.j, 'delta': -0.2}
dict_A   = {'a': 1., 'b': np.array([2,3]), 'c': 2.,}
dict_B   = {'a': 3., 'b': 5.,              'd': 2.,}
dict_C   = {         'b': np.array([8,9]), 'd': 5.,}
dict_sum = {'a': 4., 'b': np.array([15,17]), 'c': 2., 'd': 7.,}

class TestCommon(unittest.TestCase):
    def test_common(self):
        np.testing.assert_equal(conjugate_wc(wc), wc_c)
        np.testing.assert_equal(conjugate_par(par), par_c)
        np.testing.assert_equal(add_dict((dict_A, dict_B, dict_C)), dict_sum)
        np.testing.assert_equal(add_dict([dict_A, dict_B, dict_C]), dict_sum)
        np.testing.assert_equal(add_dict((dict_C, dict_A, dict_B)), dict_sum)
