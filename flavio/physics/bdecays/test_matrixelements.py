import unittest
import numpy as np
from . import rge
from . import matrixelements
from . import wilsoncoefficients
from . import matrixelements
from .. import eft, ckm
from ..running import running
from math import log
from flavio.physics.eft import WilsonCoefficients
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict

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
    ('mass','t'): 173.1,
    ('mass','c'): 1.275,
    'Gmu': 1.1663787e-5,
    'Vus': 0.22,
    'Vub': 3.7e-3,
    'Vcb': 4.1e-2,
    'gamma': 1.22,
}

class TestBMatrixElements(unittest.TestCase):
    def test_functions(self):
        # for F_8i, just see if this raises an exception
        matrixelements.F_87(0.1, 0.2, 0.3)
        matrixelements.F_89(0.4, 0.5)
        wc_obj = WilsonCoefficients()
        wc = wctot_dict(wc_obj, 'df1_bs', 4.2, par)
        matrixelements.delta_C7(par, wc, q2=3.5, scale=4.2, qiqj='bs')
        matrixelements.delta_C9(par, wc, q2=3.5, scale=4.2, qiqj='bs')
        # comparing to the values from the data file
        x =  [1.3, 0.13, 0.18]
        self.assertEqual(matrixelements.F_17(*x), -0.795182 -0.0449909j)
        self.assertEqual(matrixelements.F_19(*x),-16.3032 +0.281462j)
        self.assertEqual(matrixelements.F_27(*x), 4.77109 +0.269943j)
        self.assertEqual(matrixelements.F_29(*x), 6.75552 -1.6887j)
        # it should be that F17+F27/6=0
        self.assertAlmostEqual(matrixelements.F_17(*x),-matrixelements.F_27(*x)/6,places=5)
        # check the limiting cases of the quark loop function
        self.assertAlmostEqual(matrixelements.h(3.5, 1e-8, 4.2), matrixelements.h(3.5, 0., 4.2), places=6)
        self.assertAlmostEqual(matrixelements.h(1e-8, 1.2, 4.2), matrixelements.h(0., 1.2, 4.2), places=6)
        # comparing roughly to the plots in hep-ph/0403185v2 (but with opposite sign!)
        x = [2.5, 4.75, 4.75]
        np.testing.assert_almost_equal(-matrixelements.Fu_17(*x), 1.045 + 0.62j, decimal=1)
        np.testing.assert_almost_equal(-matrixelements.Fu_19(*x), -0.57 + 8.3j, decimal=1)
        np.testing.assert_almost_equal(-matrixelements.Fu_29(*x), -13.9 + -32.5j, decimal=0)
