import unittest
import numpy as np
from .amplitudes import *
from .observables import *
from .qcdf import *
from flavio.physics.bdecays.formfactors.b_v import bsz_parameters
from flavio.physics.eft import WilsonCoefficients
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict
from flavio.config import config
from flavio.physics.running import running
from flavio.parameters import default_parameters
import copy

s = 1.519267515435317e+24


c = copy.copy(default_parameters)
bsz_parameters.bsz_load_v1_lcsr(c)
par = c.get_central_all()

wc_obj = WilsonCoefficients()
wc = wctot_dict(wc_obj, 'bsmumu', 4.2, par)

class TestBVll(unittest.TestCase):
    def test_bksll(self):
        # just some trivial tests to see if calling the functions raises an error
        q2 = 3.5
        h = helicity_amps(q2, wc, par, 'B0', 'K*0', 'mu')
        scale = config['renormalization scale']['bvll']
        ml = par['m_mu']
        mB = par['m_B0']
        mV = par['m_K*0']
        mb = running.get_mb(par, scale)
        J = angular.angularcoeffs_general_v(h, q2, mB, mV, mb, 0, ml, ml)
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
