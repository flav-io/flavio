import unittest
import numpy as np
from .amplitudes import *
from .angulardist import *
from .observables import *
from .qcdf import *
from flavio.physics.bdecays.formfactors.b_v import bsz_parameters
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
    ('mass','t'): 173.21,
    ('mass','c'): 1.275,
    'Gmu': 1.1663787e-5,
    'Vus': 0.22,
    'Vub': 3.7e-3,
    'Vcb': 4.1e-2,
    'gamma': 1.22,
    ('f','B0'): 0.1905,
    ('f_perp','K*0'): 0.161,
    ('f_para','K*0'): 0.211,
    ('a1_perp','K*0'): 0.03,
    ('a1_para','K*0'): 0.02,
    ('a2_perp','K*0'): 0.08,
    ('a2_para','K*0'): 0.08,
}

par.update(bsz_parameters.ffpar_lcsr)

class TestBVll(unittest.TestCase):
    def test_bksll(self):
        # just some trivial tests to see if calling the functions raises an error
        q2 = 3.5
        prefactor(q2, par, 'B0', 'K*0', 'mu')
        wc_obj = WilsonCoefficients()
        wc = wctot_dict(wc_obj, 'bsmumu', 4.2, par)
        a = transversity_amps_ff(q2, wc, par, 'B0', 'K*0', 'mu')
        J = angulardist(a, q2, par, 'mu')
        # A7 should vanish as CP conjugation is ignored here (J=Jbar)
        self.assertEqual(A_experiment(J, J, 7),   0.)
        # rough numerical comparison of CP-averaged observables to 1503.05534v1
        # FIXME this should work much better with NLO corrections ...
        self.assertAlmostEqual(S_experiment(J, J, 4),  -0.151, places=1)
        self.assertAlmostEqual(S_experiment(J, J, 5),  -0.212, places=1)
        self.assertAlmostEqual(AFB_experiment(J, J),    0.002, places=1)
        self.assertAlmostEqual(FL(J, J),                0.820, places=1)
        self.assertAlmostEqual(Pp_experiment(J, J, 4), -0.413, places=1)
        self.assertAlmostEqual(Pp_experiment(J, J, 5), -0.579, places=0)
        BR = bvll_dbrdq2(q2, wc_obj, par, 'B0', 'K*0', 'mu') * 1e7
        self.assertAlmostEqual(BR, 0.467, places=1)
