import unittest
import numpy as np
from .eft import *

par = {
    ('mass','Z'): 91.1876,
    ('mass','b'): 4.18,
    ('mass','d'): 4.8e-3,
    ('mass','s'): 0.095,
    ('mass','t'): 173.3,
    ('mass','c'): 1.27,
    ('mass','u'): 2.3e-3,
    'alpha_e': 1/127.944,
    'alpha_s': 0.1185,
}

class TestEFT(unittest.TestCase):
    def test_eft(self):
        # for now just test if calling the functions raises an exception
        wc =  WilsonCoefficients()
        n_df2 = len(wc.coefficients['df2_bd'])
        n_db1 = len(wc.coefficients['df1_bs'])
        c_df2 = np.zeros(n_df2)
        c_df2[0] = 1.0
        c_db1 = np.zeros(n_db1)
        c_db1[6] = 0.3
        wc.set_initial('df2_bd', 160., c_df2)
        wc.set_initial('df1_bs', 160., c_db1)
        wc.get_wc('df2_bd', 4.2, par)
        wc.get_wc('df1_bs', 4.2, par)
