import unittest
import numpy as np
import flavio
from flavio.physics.bdecays.bxlnu import g, gLR, gVS, gVSp

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
            delta = 0.0023)
        self.assertAlmostEqual(
            flavio.sm_prediction('BR(B->Xcmunu)'),
            0.1065,
            delta = 0.0023)
        self.assertAlmostEqual(
            flavio.sm_prediction('BR(B->Xclnu)'),
            0.1065,
            delta = 0.0023)

    def test_bxclnu_lfu(self):
        self.assertAlmostEqual(
            flavio.sm_prediction('Rmue(B->Xclnu)'), 1, delta=0.01)
        self.assertAlmostEqual(
            flavio.sm_prediction('Rtaumu(B->Xclnu)'),
            flavio.sm_prediction('Rtaul(B->Xclnu)'),
            delta=0.01)
        self.assertEqual(
            flavio.physics.bdecays.bxlnu.BR_tot_leptonflavour(wc_obj, par, 'e', 'e'), 1)
        self.assertAlmostEqual(
            flavio.physics.bdecays.bxlnu.BR_tot_leptonflavour(wc_obj, par, 'tau', 'e'),
            1/flavio.physics.bdecays.bxlnu.BR_tot_leptonflavour(wc_obj, par, 'e', 'tau'))

    def test_bxclnu_np(self):
        wc_np = flavio.WilsonCoefficients()
        wc_np.set_initial({'CVR_bcenue': 0.1}, scale=4.6)
        br_sm = flavio.physics.bdecays.bxlnu.BR_BXclnu(par, wc_obj, 'e')
        br_np = flavio.physics.bdecays.bxlnu.BR_BXclnu(par, wc_np, 'e')
        # compare to the unnumbered eq. between (13) and (14) in 1407.1320
        self.assertAlmostEqual(br_np/br_sm, (1-0.34*0.1)**2, delta=0.1)

    def test_bxclnu_functions(self):
        rho = 1.2**2/4.6**2
        eps = 1e-12
        self.assertAlmostEqual(g(rho, xl=eps), g(rho, xl=0), delta=10*eps)
        self.assertAlmostEqual(gLR(rho, xl=eps), gLR(rho, xl=0), delta=10*eps)
        self.assertAlmostEqual(gVS(rho, xl=eps), 0, delta=1e-5)
        self.assertAlmostEqual(gVSp(rho, xl=eps), 0, delta=1e-5)

    def test_bxlnu_nu(self):
        wc_sm = flavio.WilsonCoefficients()
        wc_np_tau = flavio.WilsonCoefficients()
        wc_np_tau.set_initial({'CVL_bctaunutau': 1}, 4.8)
        wc_np_e = flavio.WilsonCoefficients()
        wc_np_e.set_initial({'CVL_bctaunue': 1}, 4.8)
        obs = flavio.Observable["BR(B->Xctaunu)"]
        constraints = flavio.default_parameters
        br_sm = obs.prediction_central(constraints, wc_sm)
        br_tau = obs.prediction_central(constraints, wc_np_tau)
        br_e = obs.prediction_central(constraints, wc_np_e)
        # with interference: (1 + 1)^2 = 4
        self.assertAlmostEqual(br_tau/br_sm, 4, delta=0.04)
        # without interference: 1 + 1 = 2
        self.assertAlmostEqual(br_e/br_sm, 2, delta=0.02)
