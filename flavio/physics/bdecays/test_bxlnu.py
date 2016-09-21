import unittest
import numpy as np
import flavio

par = flavio.default_parameters.get_central_all()
wc_obj = flavio.WilsonCoefficients()
par['Vcb'] = 0.04221 # inclusive Vcb: see 1411.6560

class TestBXlnu(unittest.TestCase):
    def test_bxclnu(self):
        # check that the NLO and NNLO functions reproduce the correct numbers
        self.assertAlmostEqual(
            flavio.physics.bdecays.bxlnu.pc1(r=(0.986/4.6)**2, mb=4.6),
            -1.65019, delta=0.001)
        self.assertAlmostEqual(
            flavio.physics.bdecays.bxlnu.pc2(r=(0.986/4.6)**2, mb=4.6),
            -1.91556 -0.4519 * 9 , delta=0.001)
        # check that the total BR roughly agrees with the experimental value
        # (that is present as a parameter needed for B->Xsgamma)
        self.assertAlmostEqual(
            flavio.physics.bdecays.bxlnu.BR_BXclnu(par, wc_obj, 'e')/par['BR(B->Xcenu)_exp'],
            1,
            delta = 0.05)
