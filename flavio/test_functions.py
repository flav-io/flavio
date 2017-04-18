import unittest
import flavio
from flavio.classes import Observable, Prediction

class TestFunctions(unittest.TestCase):
    def test_functions(self):
        o = Observable('test_obs')
        o.arguments = ['x']
        def f(wc_obj, par_dict, x):
            return x
        pr  = Prediction('test_obs', f )
        wc_obj = None
        self.assertEqual(flavio.sm_prediction('test_obs', 7), 7)
        self.assertEqual(flavio.np_prediction('test_obs', x=7, wc_obj=wc_obj), 7)
        self.assertEqual(flavio.sm_uncertainty('test_obs', 7), 0)
        self.assertEqual(flavio.np_uncertainty('test_obs', x=7, wc_obj=wc_obj), 0)
        self.assertEqual(flavio.sm_uncertainty('test_obs', 7, threads=2), 0)
        self.assertEqual(flavio.np_uncertainty('test_obs', x=7, wc_obj=wc_obj, threads=2), 0)
        # delete dummy instance
        Observable.del_instance('test_obs')
