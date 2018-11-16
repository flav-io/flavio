import unittest
import numpy as np
from .bpll import *
from flavio.physics.eft import WilsonCoefficients
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict
import copy
from flavio.parameters import default_parameters
from flavio.classes import Parameter
import flavio

c = default_parameters.copy()
c.set_constraint('B->K BCL a0_f+', 0.428)
c.set_constraint('B->K BCL a1_f+', -0.674)
c.set_constraint('B->K BCL a2_f+', -1.12)
c.set_constraint('B->K BCL a0_f0', 0.545)
c.set_constraint('B->K BCL a1_f0', -1.91)
c.set_constraint('B->K BCL a2_f0', 1.83)
c.set_constraint('B->K BCL a0_fT', 0.402)
c.set_constraint('B->K BCL a1_fT', -0.535)
c.set_constraint('B->K BCL a2_fT', -0.286)

par = c.get_central_all()

wc_obj = WilsonCoefficients()
wc = wctot_dict(wc_obj, 'bsmumu', 4.2, par)

wc_sm = flavio.physics.eft.WilsonCoefficients()
wc_lfv = flavio.physics.eft.WilsonCoefficients()
wc_lfv.set_initial({'C10_bsemu':4., 'C10_bsmue':2.}, 160.)


class TestBPll(unittest.TestCase):
    def test_bkll(self):
        # rough numerical test for branching ratio at high q^2 to old code
        self.assertAlmostEqual(bpll_dbrdq2(15., wc_obj, par, 'B+', 'K+', 'mu')/2.1824401629030333e-8, 1, delta=0.15)
        # test for errors
        flavio.sm_prediction('dBR/dq2(B0->Kmumu)', q2=3)
        flavio.sm_prediction('AFB(B0->Kmumu)', q2=15)
        flavio.sm_prediction('FH(B+->Kmumu)', q2=21)
        # direct CP asymmetry should be close to 0
        self.assertAlmostEqual(flavio.sm_prediction("ACP(B0->Kmumu)", q2=1), 0, delta=0.01)
        self.assertAlmostEqual(flavio.sm_prediction("ACP(B0->Kmumu)", q2=6), 0, delta=0.01)
        self.assertAlmostEqual(flavio.sm_prediction("ACP(B0->Kmumu)", q2=17), 0, delta=0.01)
        self.assertAlmostEqual(flavio.sm_prediction("ACP(B+->Kmumu)", q2=1), 0, delta=0.01)
        self.assertAlmostEqual(flavio.sm_prediction("ACP(B+->Kmumu)", q2=6), 0, delta=0.01)
        self.assertAlmostEqual(flavio.sm_prediction("ACP(B+->Kmumu)", q2=17), 0, delta=0.01)
        # LFU ratio = 1
        self.assertAlmostEqual(flavio.sm_prediction("Rmue(B+->Kll)", q2=6), 1, delta=1e-3)
        self.assertAlmostEqual(flavio.sm_prediction("Rmue(B0->Kll)", q2=6), 1, delta=1e-3)

    def test_bktautau(self):
        brbin = flavio.sm_prediction('<dBR/dq2>(B+->Ktautau)', 12, 24)
        brtot = flavio.sm_prediction('BR(B+->Ktautau)')
        self.assertAlmostEqual(brtot / brbin, 12, delta=0.1)
