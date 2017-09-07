import unittest
import numpy as np
import numpy.testing as npt
import flavio

def f(x):
    return (x[0]-2)**2 + (x[1]-1)**2
def g(x):
    return -f(x)
def h(x, a):
    return (x[0]-a)**2 + (x[1]-1)**2

class TestOptimize(unittest.TestCase):
    def test_slsqp(self):
        res = flavio.math.optimize.minimize_robust(f, [0, 0], disp=False, methods=('SLSQP',))
        npt.assert_array_almost_equal(res.x, [2, 1])
        res = flavio.math.optimize.maximize_robust(g, [5, 5], disp=False, methods=('SLSQP',))
        npt.assert_array_almost_equal(res.x, [2, 1])
        res = flavio.math.optimize.minimize_robust(h, [0, 0], args=(3,), methods=('SLSQP',))
        npt.assert_array_almost_equal(res.x, [3, 1])

    def test_minuit(self):
        res = flavio.math.optimize.minimize_migrad(f, [0, 0], print_level=0)
        npt.assert_array_almost_equal(res.x, [2, 1])
        res = flavio.math.optimize.minimize_robust(f, [0, 0], methods=('MIGRAD',))
        npt.assert_array_almost_equal(res.x, [2, 1])
        res = flavio.math.optimize.minimize_robust(h, [0, 0], args=(3,), methods=('MIGRAD',))
        npt.assert_array_almost_equal(res.x, [3, 1])
