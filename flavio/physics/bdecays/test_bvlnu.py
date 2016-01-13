import unittest
import numpy as np
from .bvlnu import *
from flavio.physics.bdecays.formfactors.b_v import bsz_parameters


s = 1.519267515435317e+24

par = {
    ('mass','e'): 0.510998928e-3,
    ('mass','mu'): 105.6583715e-3,
    ('mass','tau'): 1.77686,
    ('mass','B+'): 5.27929,
    ('mass','B0'): 5.27961,
    ('mass','rho+'): 0.077526,
    ('mass','rho0'): 0.077526,
    ('lifetime','B+'): 1638.e-15*s,
    ('lifetime','B0'): 152.e-14*s,
    ('mass','b','MSbar'): 4.18,
    'Gmu': 1.1663787e-5,
    'alphaem': 1/127.940,
    'Vus': 0.22,
    'Vub': 3.7e-3,
    'Vcb': 4.1e-2,
    'gamma': 1.22,
}

par.update(bsz_parameters.ffpar_lcsr)

class TestBVll(unittest.TestCase):
    def test_bksll(self):
        # just some trivial tests to see if calling the functions raises an error
        q2 = 3.5
        helicity_amps(q2, par, 'B0', 'rho+', 'e')
        dBR(q2, par, 'B0', 'rho+', 'e')
