import unittest
import numpy as np
import flavio

constraints = flavio.default_parameters
wc_obj = flavio.WilsonCoefficients()
par = constraints.get_central_all()

class TestBlnu(unittest.TestCase):
    def test_blnu(self):
        Vub = flavio.physics.ckm.get_ckm(par)[0,2]
        # compare to literature value
        self.assertAlmostEqual(
            flavio.Observable["BR(B+->taunu)"].prediction_central(constraints, wc_obj),
            1.1e-4 * (abs(Vub)/3.95e-3)**2 * (par['f_B+']/0.2)**2,
            delta=2e-6)
        # check that B->enu BR is smaller than B->munu
        # (ratio given by mass ratio squared)
        self.assertAlmostEqual(
            (
            flavio.Observable["BR(B+->enu)"].prediction_central(constraints, wc_obj)/
            flavio.Observable["BR(B+->munu)"].prediction_central(constraints, wc_obj)
            )/(par['m_e']**2/par['m_mu']**2),
            1,
            delta=0.001) # there are corrections of order mmu**2/mB**2
        # check that Bc->taunu is larger by Vcb^2/Vub^2 (and more due to decay constant)
        self.assertAlmostEqual(
            (flavio.Observable["BR(Bc->taunu)"].prediction_central(constraints, wc_obj) / 0.04**2)
            /
            (flavio.Observable["BR(B+->taunu)"].prediction_central(constraints, wc_obj) / 0.0036**2),
            3,
            delta=1.5  # very rough
            )

    def test_blnu_nu(self):
        wc_sm = flavio.WilsonCoefficients()
        wc_np_tau = flavio.WilsonCoefficients()
        wc_np_tau.set_initial({'CVL_butaunutau': 1}, 4.8)
        wc_np_e = flavio.WilsonCoefficients()
        wc_np_e.set_initial({'CVL_butaunue': 1}, 4.8)
        obs = flavio.Observable["BR(B+->taunu)"]
        br_sm = obs.prediction_central(constraints, wc_sm)
        br_tau = obs.prediction_central(constraints, wc_np_tau)
        br_e = obs.prediction_central(constraints, wc_np_e)
        # with interference: (1 + 1)^2 = 4
        self.assertAlmostEqual(br_tau/br_sm, 4, delta=0.04)
        # without interference: 1 + 1 = 2
        self.assertAlmostEqual(br_e/br_sm, 2, delta=0.02)
