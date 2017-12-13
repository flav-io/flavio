import unittest
import numpy as np
import flavio

constraints = flavio.default_parameters
wc_obj = flavio.WilsonCoefficients()
par = constraints.get_central_all()

class TestKpilnu(unittest.TestCase):
    def test_kpilnu(self):
        # test for errors
        q2=0.05
        flavio.physics.kdecays.kpilnu.get_ff(q2, par, 'KL')
        flavio.physics.kdecays.kpilnu.get_ff(q2, par, 'K+')
        flavio.physics.kdecays.kpilnu.get_angularcoeff(q2, wc_obj, par, 'KL', 'pi+', 'e')
        flavio.physics.kdecays.kpilnu.get_angularcoeff(q2, wc_obj, par, 'KL', 'pi+', 'mu')
        flavio.physics.kdecays.kpilnu.get_angularcoeff(q2, wc_obj, par, 'K+', 'pi0', 'e')
        flavio.physics.kdecays.kpilnu.get_angularcoeff(q2, wc_obj, par, 'K+', 'pi0', 'mu')
        # unphysical q2
        self.assertEqual(flavio.physics.kdecays.kpilnu.dBRdq2(0, wc_obj, par, 'KL', 'pi+', 'e'), 0)
        self.assertEqual(flavio.physics.kdecays.kpilnu.dBRdq2(0.01, wc_obj, par, 'K+', 'pi0', 'mu'), 0)
        self.assertEqual(flavio.physics.kdecays.kpilnu.dBRdq2(2, wc_obj, par, 'K+', 'pi0', 'e'), 0)
        self.assertEqual(flavio.physics.kdecays.kpilnu.dBRdq2(1.5, wc_obj, par, 'KL', 'pi+', 'mu'), 0)
        self.assertEqual(flavio.physics.kdecays.kpilnu.dBRdq2(0.1, wc_obj, par, 'KL', 'pi+', 'tau'), 0)
        # compare central predictions to PDG values
        self.assertAlmostEqual(flavio.sm_prediction('BR(KL->pienu)')*1e2/40.55, 1, delta=0.04)
        self.assertAlmostEqual(flavio.sm_prediction('BR(K+->pienu)')*1e2/5.07, 1, delta=0.04)
        self.assertAlmostEqual(flavio.sm_prediction('BR(KL->pimunu)')*1e2/27.04, 1, delta=0.02)
        self.assertAlmostEqual(flavio.sm_prediction('BR(K+->pimunu)')*1e2/3.352, 1, delta=0.03)


    def test_kpilnu_nu(self):
        wc_sm = flavio.WilsonCoefficients()
        wc_np_mu = flavio.WilsonCoefficients()
        wc_np_mu.set_initial({'CVL_sumunumu': 1}, 4.8)
        wc_np_e = flavio.WilsonCoefficients()
        wc_np_e.set_initial({'CVL_sumunue': 1}, 4.8)
        obs = flavio.Observable["BR(K+->pimunu)"]
        br_sm = obs.prediction_central(constraints, wc_sm)
        br_mu = obs.prediction_central(constraints, wc_np_mu)
        br_e = obs.prediction_central(constraints, wc_np_e)
        # with interference: (1 + 1)^2 = 4
        self.assertAlmostEqual(br_mu/br_sm, 4, delta=0.06)
        # without interference: 1 + 1 = 2
        self.assertAlmostEqual(br_e/br_sm, 2, delta=0.03)
