import unittest
import flavio
from flavio.physics.bdecays import lifetime_ratio
import numpy as np

par = flavio.default_parameters.get_central_all()
wc_sm = flavio.WilsonCoefficients()


class TestTauBpoBd(unittest.TestCase):
    def test_bag_params(self):
        flavio_bag_params = lifetime_ratio.run_lifetime_bag_parameters(par, 4.5)
        # These are the results at 4.5 GeV directly from Maria Laura Piscopo
        MLP_results = {
            "bag_lifetime_B1qtilde":  1.0080976022,
            "bag_lifetime_B2qtilde":  1.0012180386,
            "bag_lifetime_B3qtilde": -0.04773286535,
            "bag_lifetime_B4qtilde": -0.03431299826,
            "bag_lifetime_B5qtilde": -1.0,
            "bag_lifetime_B6qtilde": -1.0,
            "bag_lifetime_B7qtilde":  0.0,
            "bag_lifetime_B8qtilde":  0.0,
            "bag_lifetime_deltaqq1tilde":  0.0026,
            "bag_lifetime_deltaqq2tilde": -0.0018,
            "bag_lifetime_deltaqq3tilde": -0.0004,
            "bag_lifetime_deltaqq4tilde":  0.0003,
        }
        for bag_param in flavio_bag_params:
            self.assertAlmostEqual(flavio_bag_params[bag_param], MLP_results[bag_param])

    def test_sm(self):
        self.assertAlmostEqual(flavio.sm_prediction("tau_B+/tau_Bd"), 1.08381512025)
        self.assertAlmostEqual(flavio.sm_uncertainty("tau_B+/tau_Bd", N=1000), 0.016, delta=0.002)

    def test_A_WE_cu(self):
        me = {}
        for i in range(1, 9):
            me[f"{i}"]  = i
            me[f"{i}p"] = i
        A_WE_cu = np.array(
            (
                (3.75, 1.35, -8.222, -1.897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.575, -0.975, -4.822, -1.66, -49.015, -16.128),
                (1.35, 4.05, -1.897, -5.692, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.975, -2.925, -1.66, -4.981, -16.128, -48.383),
                (-8.222, -1.897, 52.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.538, 2.846, 27.0, 9.0, 324.0, 108.0),
                (-1.897, -5.692, 12.0, 36.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.846, 8.538, 9.0, 27.0, 108.0, 324.0),
                (0.0, 0.0, 0.0, 0.0, 0.937, 0.337, 1.304, 0.356, 11.226, 2.372, -2.575, -0.975, 8.538, 2.846, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 0.337, 1.012, 0.356, 1.067, 2.372, 7.115, -0.975, -2.925, 2.846, 8.538, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 1.304, 0.356, 3.6, 0.9, 37.6, 8.4, -4.822, -1.66, 27.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 0.356, 1.067, 0.9, 2.7, 8.4, 25.2, -1.66, -4.981, 9.0, 27.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 11.226, 2.372, 37.6, 8.4, 473.6, 110.4, -49.015, -16.128, 324.0, 108.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 2.372, 7.115, 8.4, 25.2, 110.4, 331.2, -16.128, -48.383, 108.0, 324.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, -2.575, -0.975, -4.822, -1.66, -49.015, -16.128, 3.75, 1.35, -8.222, -1.897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, -0.975, -2.925, -1.66, -4.981, -16.128, -48.383, 1.35, 4.05, -1.897, -5.692, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 8.538, 2.846, 27.0, 9.0, 324.0, 108.0, -8.222, -1.897, 52.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 2.846, 8.538, 9.0, 27.0, 108.0, 324.0, -1.897, -5.692, 12.0, 36.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (-2.575, -0.975, 8.538, 2.846, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.937, 0.337, 1.304, 0.356, 11.226, 2.372),
                (-0.975, -2.925, 2.846, 8.538, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.337, 1.012, 0.356, 1.067, 2.372, 7.115),
                (-4.822, -1.66, 27.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.304, 0.356, 3.6, 0.9, 37.6, 8.4),
                (-1.66, -4.981, 9.0, 27.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.356, 1.067, 0.9, 2.7, 8.4, 25.2),
                (-49.015, -16.128, 324.0, 108.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.226, 2.372, 37.6, 8.4, 473.6, 110.4),
                (-16.128, -48.383, 108.0, 324.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.372, 7.115, 8.4, 25.2, 110.4, 331.2),
            )
        )
        np.testing.assert_allclose(lifetime_ratio.A_WE_cu(0.1, me), A_WE_cu, atol=1e-3)

    def test_A_WE_cc(self):
        me = {}
        for i in range(1, 9):
            me[f"{i}"]  = i
            me[f"{i}p"] = i
        A_WE_cc = np.array(
            (
                (4.7, 1.5, -8.222, -1.897, 0.95, 0.15, 1.502, 0.237, 26.879, 6.641, -4.7, -1.5, 17.076, 5.692, -3.75, -1.35, -4.822, -1.66, -49.015, -16.128),
                (1.5, 4.5, -1.897, -5.692, 0.15, 0.45, 0.237, 0.712, 6.641, 19.922, -1.5, -4.5, 5.692, 17.076, -1.35, -4.05, -1.66, -4.981, -16.128, -48.383),
                (-8.222, -1.897, 41.6, 9.6, -4.111, -0.949, -2.6, -0.6, -31.2, -7.2, 17.076, 5.692, -21.6, -7.2, 8.538, 2.846, 21.6, 7.2, 259.2, 86.4),
                (-1.897, -5.692, 9.6, 28.8, -0.949, -2.846, -0.6, -1.8, -7.2, -21.6, 5.692, 17.076, -7.2, -21.6, 2.846, 8.538, 7.2, 21.6, 86.4, 259.2),
                (0.95, 0.15, -4.111, -0.949, 1.175, 0.375, 1.304, 0.356, 11.226, 2.372, -3.75, -1.35, 8.538, 2.846, -1.175, -0.375, -1.858, -0.593, -26.721, -9.012),
                (0.15, 0.45, -0.949, -2.846, 0.375, 1.125, 0.356, 1.067, 2.372, 7.115, -1.35, -4.05, 2.846, 8.538, -0.375, -1.125, -0.593, -1.779, -9.012, -27.037),
                (1.502, 0.237, -2.6, -0.6, 1.304, 0.356, 2.95, 0.75, 29.8, 6.6, -4.822, -1.66, 21.6, 7.2, -1.858, -0.593, -1.175, -0.375, -16.9, -5.7),
                (0.237, 0.712, -0.6, -1.8, 0.356, 1.067, 0.75, 2.25, 6.6, 19.8, -1.66, -4.981, 7.2, 21.6, -0.593, -1.779, -0.375, -1.125, -5.7, -17.1),
                (26.879, 6.641, -31.2, -7.2, 11.226, 2.372, 29.8, 6.6, 380.0, 88.8, -49.015, -16.128, 259.2, 86.4, -26.721, -9.012, -16.9, -5.7, -191.6, -63.6),
                (6.641, 19.922, -7.2, -21.6, 2.372, 7.115, 6.6, 19.8, 88.8, 266.4, -16.128, -48.383, 86.4, 259.2, -9.012, -27.037, -5.7, -17.1, -63.6, -190.8),
                (-4.7, -1.5, 17.076, 5.692, -3.75, -1.35, -4.822, -1.66, -49.015, -16.128, 4.7, 1.5, -8.222, -1.897, 0.95, 0.15, 1.502, 0.237, 26.879, 6.641),
                (-1.5, -4.5, 5.692, 17.076, -1.35, -4.05, -1.66, -4.981, -16.128, -48.383, 1.5, 4.5, -1.897, -5.692, 0.15, 0.45, 0.237, 0.712, 6.641, 19.922),
                (17.076, 5.692, -21.6, -7.2, 8.538, 2.846, 21.6, 7.2, 259.2, 86.4, -8.222, -1.897, 41.6, 9.6, -4.111, -0.949, -2.6, -0.6, -31.2, -7.2),
                (5.692, 17.076, -7.2, -21.6, 2.846, 8.538, 7.2, 21.6, 86.4, 259.2, -1.897, -5.692, 9.6, 28.8, -0.949, -2.846, -0.6, -1.8, -7.2, -21.6),
                (-3.75, -1.35, 8.538, 2.846, -1.175, -0.375, -1.858, -0.593, -26.721, -9.012, 0.95, 0.15, -4.111, -0.949, 1.175, 0.375, 1.304, 0.356, 11.226, 2.372),
                (-1.35, -4.05, 2.846, 8.538, -0.375, -1.125, -0.593, -1.779, -9.012, -27.037, 0.15, 0.45, -0.949, -2.846, 0.375, 1.125, 0.356, 1.067, 2.372, 7.115),
                (-4.822, -1.66, 21.6, 7.2, -1.858, -0.593, -1.175, -0.375, -16.9, -5.7, 1.502, 0.237, -2.6, -0.6, 1.304, 0.356, 2.95, 0.75, 29.8, 6.6),
                (-1.66, -4.981, 7.2, 21.6, -0.593, -1.779, -0.375, -1.125, -5.7, -17.1, 0.237, 0.712, -0.6, -1.8, 0.356, 1.067, 0.75, 2.25, 6.6, 19.8),
                (-49.015, -16.128, 259.2, 86.4, -26.721, -9.012, -16.9, -5.7, -191.6, -63.6, 26.879, 6.641, -31.2, -7.2, 11.226, 2.372, 29.8, 6.6, 380.0, 88.8),
                (-16.128, -48.383, 86.4, 259.2, -9.012, -27.037, -5.7, -17.1, -63.6, -190.8, 6.641, 19.922, -7.2, -21.6, 2.372, 7.115, 6.6, 19.8, 88.8, 266.4),
            )
        )
        np.testing.assert_allclose(lifetime_ratio.A_WE_cc(0.1, me), A_WE_cc, atol=1e-3)

    def test_A_PI_cd(self):
        me = {}
        for i in range(1, 9):
            me[f"{i}"]  = i
            me[f"{i}p"] = i
        A_PI_cd = np.array(
            (
                (19.0, 3.0, -3.004, -0.474, 4.822, 1.66, 15.25, 5.25, -183.0, -63.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (3.0, 19.0, -0.474, -3.004, 1.66, 4.822, 5.25, 15.25, -63.0, -183.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (-3.004, -0.474, -14.4, -3.6, -14.325, -4.725, -8.538, -2.846, 4.427, 1.897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (-0.474, -3.004, -3.6, -14.4, -4.725, -14.325, -2.846, -8.538, 1.897, 4.427, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (4.822, 1.66, -14.325, -4.725, -3.6, -0.9, -1.304, -0.356, -11.226, -2.372, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (1.66, 4.822, -4.725, -14.325, -0.9, -3.6, -0.356, -1.304, -2.372, -11.226, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (15.25, 5.25, -8.538, -2.846, -1.304, -0.356, -0.937, -0.337, -22.75, -4.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (5.25, 15.25, -2.846, -8.538, -0.356, -1.304, -0.337, -0.937, -4.35, -22.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (-183.0, -63.0, 4.427, 1.897, -11.226, -2.372, -22.75, -4.35, 137.0, 18.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (-63.0, -183.0, 1.897, 4.427, -2.372, -11.226, -4.35, -22.75, 18.6, 137.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 19.0, 3.0, -3.004, -0.474, 4.822, 1.66, 15.25, 5.25, -183.0, -63.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 19.0, -0.474, -3.004, 1.66, 4.822, 5.25, 15.25, -63.0, -183.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.004, -0.474, -14.4, -3.6, -14.325, -4.725, -8.538, -2.846, 4.427, 1.897),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.474, -3.004, -3.6, -14.4, -4.725, -14.325, -2.846, -8.538, 1.897, 4.427),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.822, 1.66, -14.325, -4.725, -3.6, -0.9, -1.304, -0.356, -11.226, -2.372),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.66, 4.822, -4.725, -14.325, -0.9, -3.6, -0.356, -1.304, -2.372, -11.226),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.25, 5.25, -8.538, -2.846, -1.304, -0.356, -0.937, -0.337, -22.75, -4.35),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.25, 15.25, -2.846, -8.538, -0.356, -1.304, -0.337, -0.937, -4.35, -22.75),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -183.0, -63.0, 4.427, 1.897, -11.226, -2.372, -22.75, -4.35, 137.0, 18.6),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -63.0, -183.0, 1.897, 4.427, -2.372, -11.226, -4.35, -22.75, 18.6, 137.0),
            )
        )
        np.testing.assert_allclose(lifetime_ratio.A_PI_cd(0.1, me), A_PI_cd, atol=1e-3)

    def test_WE_cu(self):
        self.assertAlmostEqual(lifetime_ratio.weak_exchange(wc_sm, par, "B0"), 0, delta=1e-25)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLL_bcud": 1}, scale=4.5)
        self.assertAlmostEqual(1e15 * lifetime_ratio.weak_exchange(wc, par, "B0"), 3.03733567, places=6)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLLt_bcud": 1}, scale=4.5)
        self.assertAlmostEqual(1e15 * lifetime_ratio.weak_exchange(wc, par, "B0"), 9.79346592, places=5)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLLt_bcus": 1}, scale=4.5)
        self.assertAlmostEqual(lifetime_ratio.weak_exchange(wc, par, "B0"), 0, delta=1e-25)

    def test_PI_cd(self):
        self.assertAlmostEqual(lifetime_ratio.pauli_interference(wc_sm, par, "B+"), 0, delta=1e-25)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLL_bcud": 1}, scale=4.5)
        self.assertAlmostEqual(1e14 * lifetime_ratio.pauli_interference(wc, par, "B+"), 2.82494420, places=6)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLL_bcus": 1}, scale=4.5)
        self.assertAlmostEqual(1e14 * lifetime_ratio.pauli_interference(wc, par, "B+"), 0.1503590699, places=6)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLLt_bcud": 1}, scale=4.5)
        self.assertAlmostEqual(1e13 * lifetime_ratio.pauli_interference(wc, par, "B+"), 1.38771099, places=6)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLLt_bcus": 1}, scale=4.5)
        Vud = flavio.physics.ckm.get_ckm(par)[0,0]
        Vus = flavio.physics.ckm.get_ckm(par)[0,1]
        self.assertAlmostEqual(1e13 * lifetime_ratio.pauli_interference(wc, par, "B+"), 1.38771099 * (Vus / Vud) ** 2, places=6)

    def test_NP_bcud(self):
        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLL_bcud": -2}, scale=4.5)
        self.assertAlmostEqual(flavio.np_prediction("tau_B+/tau_Bd", wc), 1.011517673)
        self.assertAlmostEqual(flavio.np_uncertainty("tau_B+/tau_Bd", wc, N=1000), 0.013, delta=0.002)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLLt_bcud": 3}, scale=4.5)
        self.assertAlmostEqual(flavio.np_prediction("tau_B+/tau_Bd", wc), 0.007740969, places=6)
        self.assertAlmostEqual(flavio.np_uncertainty("tau_B+/tau_Bd", wc, N=1000), 0.082, delta=0.006)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVRLt_bcud": -0.15, "CSLL_bcud": 0.2}, scale=4.5)
        self.assertAlmostEqual(flavio.np_prediction("tau_B+/tau_Bd", wc), 1.113086143)
        self.assertAlmostEqual(flavio.np_uncertainty("tau_B+/tau_Bd", wc, N=1000), 0.016, delta=0.002)

    def test_NP_bcus(self):
        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLL_bcus": -2}, scale=4.5)
        self.assertAlmostEqual(flavio.np_prediction("tau_B+/tau_Bd", wc), 1.079837520)
        self.assertAlmostEqual(flavio.np_uncertainty("tau_B+/tau_Bd", wc, N=1000), 0.016, delta=0.002)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLLt_bcus": 3}, scale=4.5)
        self.assertAlmostEqual(flavio.np_prediction("tau_B+/tau_Bd", wc), 1.017205203)
        self.assertAlmostEqual(flavio.np_uncertainty("tau_B+/tau_Bd", wc, N=1000), 0.019, delta=0.002)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVRLt_bcus": -0.15, "CSLL_bcus": 0.2}, scale=4.5)
        self.assertAlmostEqual(flavio.np_prediction("tau_B+/tau_Bd", wc), 1.083496791)
        self.assertAlmostEqual(flavio.np_uncertainty("tau_B+/tau_Bd", wc, N=1000), 0.016, delta=0.002)

    def test_NP_dbcc(self):
        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLL_bdcc": -2}, scale=4.5)
        self.assertAlmostEqual(flavio.np_prediction("tau_B+/tau_Bd", wc), 1.084098486)
        self.assertAlmostEqual(flavio.np_uncertainty("tau_B+/tau_Bd", wc, N=1000), 0.016, delta=0.002)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLLt_bdcc": 3}, scale=4.5)
        self.assertAlmostEqual(flavio.np_prediction("tau_B+/tau_Bd", wc), 1.102675260)
        self.assertAlmostEqual(flavio.np_uncertainty("tau_B+/tau_Bd", wc, N=1000), 0.016, delta=0.002)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVRLt_bdcc": -0.15, "CSLL_bdcc": 0.2}, scale=4.5)
        self.assertAlmostEqual(flavio.np_prediction("tau_B+/tau_Bd", wc), 1.085435224)
        self.assertAlmostEqual(flavio.np_uncertainty("tau_B+/tau_Bd", wc, N=1000), 0.016, delta=0.002)
