import unittest
import numpy as np
from .qcdf import *
from flavio.physics.eft import WilsonCoefficients
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict
from flavio.physics.bdecays.formfactors.b_v import bsz_parameters

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
    ('f','B0'): 0.1905,
    ('f_perp','K*0'): 0.185,
    ('f_para','K*0'): 0.225,
    ('a1_perp','K*0'): 0.2,
    ('a1_para','K*0'): 0.2,
    ('a2_perp','K*0'): 0.05,
    ('a2_para','K*0'): 0.05,
}

par.update(bsz_parameters.ffpar_lcsr)

wc_obj = WilsonCoefficients()
wc = wctot_dict(wc_obj, 'df1_bs', 4.2, par)

class TestQCDF(unittest.TestCase):
    def test_qcdf(self):
        # just some trivial tests to see if calling the functions raises an error
        q2=3.5
        B='B0'
        V='K*0'
        T_para(q2, par, wc, B, V, scale=4.2)
        T_perp(q2, par, wc, B, V)
