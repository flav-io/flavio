import unittest
import numpy as np
from . import rge


s = 1.519267515435317e+24

par = {
    ('mass','e'): 0.510998928e-3,
    ('mass','mu'): 105.6583715e-3,
    ('mass','tau'): 1.77686,
    ('mass','B+'): 5.27929,
    ('mass','B0'): 5.27961,
    ('mass','Bs'): 5.36679,
    ('mass','K*0'): 0.89166,
    ('lifetime','B+'): 1638.e-15*s,
    ('lifetime','B0'): 152.e-14*s,
    'alpha_e': 1/127.940,
    'alpha_s': 0.1185,
    ('mass','Z'): 91.1876,
    ('mass','b'): 4.17,
    ('mass','t'): 173.21,
    ('mass','c'): 1.275,
    'Gmu': 1.1663787e-5,
    'Vus': 0.22,
    'Vub': 3.7e-3,
    'Vcb': 4.1e-2,
    'gamma': 1.22,
}


wc = {
    'C7': 0,
    'C7p': 0,
    'C9': 0,
    'C9p': 0,
    'C10': 0,
    'C10p': 0,
    'CP': 0,
    'CPp': 0,
    'CS': 0,
    'CSp': 0,
}

class TestBWilson(unittest.TestCase):
    def test_running(self):
        c_in = np.array([ 0.85143759,  0.31944853,  0.30029457,  0.82914154,  0.11154786,
        0.80629828,  0.32082766,  0.1300508 ,  0.69393572,  0.98427495,
        0.76415058,  0.90545245,  0.03290275,  0.89359186,  0.46273251])
        c_out = rge.run_wc_df1(par, c_in, 173.2, 4.2)
        print(c_out/c_in)
