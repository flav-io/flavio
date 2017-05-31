import unittest
import numpy as np
import numpy.testing as npt
import flavio

class TestOptimize(unittest.TestCase):
    def test_optimize(self):
        def f(x):
            return (x[0]-2)**2 + (x[1]-1)**2
        def g(x):
            return -f(x)
        res = flavio.math.optimize.minimize_robust(f, [0, 0], disp=True)
        npt.assert_array_almost_equal(res.x, [2, 1])
        res = flavio.math.optimize.maximize_robust(g, [5, 5], disp=True)
        npt.assert_array_almost_equal(res.x, [2, 1])
