import unittest
import numpy as np
from . import amplitude
from math import sin
from cmath import phase

s = 1.519267515435317e+24

par = {
    ('mass','B0'): 5.27961,
    ('mass','Bs'): 5.36679,
    ('mass','b'): 4.17,
    ('mass','d'): 4.8e-3,
    ('mass','s'): 0.095,
    ('mass','t'): 173.21,
    ('mass','c'): 1.275,
    ('mass','u'): 2.3e-3,
    ('mass','W'): 80.4,
    ('lifetime','B0'): 152.e-14*s,
    ('f','B0'): 0.1905,
    ('bag','B0',1): 1.27/1.517,
    ('bag','B0',4): 0.95,
    ('bag','B0',5): 1.47,
    ('f','Bs'): 0.2277,
    ('bag','Bs',1): 1.33/1.517,
    ('bag','Bs',4): 0.93,
    ('bag','Bs',5): 1.57,
    'Gmu': 1.1663787e-5,
    'alphaem': 1/127.940,
    'Vus': 0.22,
    'Vub': 3.7e-3,
    'Vcb': 4.1e-2,
    'gamma': 1.22,
    'alpha_s': 0.1185,
    ('eta_tt', 'B0'): 0.55,
    ('eta_tt', 'Bs'): 0.55,
    ('eta_tt', 'K0'): 0.57,
    ('eta_cc', 'K0'): 1.38,
    ('eta_ct', 'K0'): 0.47,
}

wc = {
    'CVLL': 0,
}


class TestMesonMixing(unittest.TestCase):
    def test_bmixing(self):
        # just some trivial tests to see if calling the functions raises an error
        m12d = amplitude.M12_d(par, wc, 'B0')
        m12s = amplitude.M12_d(par, wc, 'Bs')
