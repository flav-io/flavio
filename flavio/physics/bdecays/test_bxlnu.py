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
        self.assertAlmostEqual(
            flavio.sm_prediction('BR(B->Xcenu)'),
            0.1065,
            delta = 0.0005)
        self.assertAlmostEqual(
            flavio.sm_prediction('BR(B->Xcmunu)'),
            0.1065,
            delta = 0.0005)
        self.assertAlmostEqual(
            flavio.sm_prediction('BR(B->Xclnu)'),
            0.1065,
            delta = 0.0005)

    def test_bxclnu_np(self):
        wc_np = flavio.WilsonCoefficients()
        wc_np.set_initial({'CVp_bcenu': 0.1}, scale=4.6)
        br_sm = flavio.physics.bdecays.bxlnu.BR_BXclnu(par, wc_obj, 'e')
        br_np = flavio.physics.bdecays.bxlnu.BR_BXclnu(par, wc_np, 'e')
        # compare to the unnumbered eq. between (13) and (14) in 1407.1320
        self.assertAlmostEqual(br_np/br_sm, (1-0.34*0.1)**2, delta=0.1)
