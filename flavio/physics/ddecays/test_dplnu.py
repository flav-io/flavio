import unittest
import flavio


class TestDPll(unittest.TestCase):

    def test_dplnu_exp(self):
        self.assertAlmostEqual(flavio.sm_prediction('BR(D0->Kenu)'),
                               3.530e-2, delta=0.3e-2)
        self.assertAlmostEqual(flavio.sm_prediction('BR(D0->Kmunu)'),
                               3.31e-2, delta=0.3e-2)
        self.assertAlmostEqual(flavio.sm_prediction('BR(D0->pienu)'),
                               2.91e-3, delta=3 * 0.3e-3)
        self.assertAlmostEqual(flavio.sm_prediction('BR(D0->pimunu)'),
                               2.37e-3, delta=0.4e-3)
        self.assertAlmostEqual(flavio.sm_prediction('BR(D+->Kenu)'),
                               8.73e-2, delta=0.8e-2)
        self.assertAlmostEqual(flavio.sm_prediction('BR(D+->Kmunu)'),
                               8.74e-2, delta=0.8e-2)
        self.assertAlmostEqual(flavio.sm_prediction('BR(D+->pienu)'),
                               3.72e-3, delta=2 * 0.4e-3)


    def test_dplnu_nu(self):
        wc_sm = flavio.WilsonCoefficients()
        wc_np_mu = flavio.WilsonCoefficients()
        wc_np_mu.set_initial({'CVL_dcmunumu': 1}, 2)
        wc_np_e = flavio.WilsonCoefficients()
        wc_np_e.set_initial({'CVL_dcmunue': 1}, 2)
        obs = flavio.Observable["BR(D+->pimunu)"]
        constraints = flavio.default_parameters
        br_sm = obs.prediction_central(constraints, wc_sm)
        br_mu = obs.prediction_central(constraints, wc_np_mu)
        br_e = obs.prediction_central(constraints, wc_np_e)
        # with interference: (1 + 1)^2 = 4
        self.assertAlmostEqual(br_mu / br_sm, 4, delta=0.04)
        # without interference: 1 + 1 = 2
        self.assertAlmostEqual(br_e / br_sm, 2, delta=0.02)
