import unittest
import numpy as np
from .eft import *

par = {
    'm_Z': 91.1876,
    'm_b': 4.18,
    'm_d': 4.8e-3,
    'm_s': 0.095,
    'm_t': 173.3,
    'm_c': 1.27,
    'm_u': 2.3e-3,
    'alpha_e': 1/127.944,
    'alpha_s': 0.1185,
}

class TestEFT(unittest.TestCase):
    def test_eft(self):
        wc =  WilsonCoefficients()
        wc.set_initial({'CVLL_bsbs': 0.1j, 'C9_bsmumu':-1.5, 'CV_bctaunu': 0.2}, 160.)
        wc.get_wc('bsbs', 4.8, par)
        wc.get_wc('bsmumu', 4.8, par)
        wc.get_wc('bctaunu', 4.8, par)
