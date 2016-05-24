import unittest
import numpy as np
import flavio
import math

s = 1.519267515435317e+24
ps = 1e-12*s


class TestIntegrate(unittest.TestCase):
    def test_nintegrate(self):
        # integrate sin(x) from 0 to 2
        xmin = 0
        xmax = 2
        val = 2*math.sin(1)**2
        self.assertAlmostEqual(flavio.math.integrate.nintegrate(math.sin, xmin, xmax), val, delta=0.01*val)
        self.assertAlmostEqual(flavio.math.integrate.nintegrate_fast(math.sin, xmin, xmax), val, delta=0.01*val)
