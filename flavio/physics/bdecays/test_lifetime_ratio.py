import unittest
import flavio


par = flavio.default_parameters.get_central_all()

wc_sm = flavio.WilsonCoefficients()

class TestTauBpoBd(unittest.TestCase):
    def test_bag_params(self):
        flavio_bag_params = flavio.physics.bdecays.lifetime_ratio.run_lifetime_bag_parameters(par, 4.5)
        MLP_results = {
            "bag_lifetime_B1qtilde": 1.0080976022,
            "bag_lifetime_B2qtilde": 1.0012180386,
            "bag_lifetime_B3qtilde": -0.04773286535,
            "bag_lifetime_B4qtilde": -0.03431299826,
            "bag_lifetime_B5qtilde": -1.0,
            "bag_lifetime_B6qtilde": -1.0,
            "bag_lifetime_B7qtilde": 0.0,
            "bag_lifetime_B8qtilde": 0.0,
            "bag_lifetime_deltaqq1tilde":  0.0026,
            "bag_lifetime_deltaqq2tilde": -0.0018,
            "bag_lifetime_deltaqq3tilde": -0.0004,
            "bag_lifetime_deltaqq4tilde": 0.0003,
        }
        for bag_param in flavio_bag_params:
            self.assertAlmostEqual(flavio_bag_params[bag_param], MLP_results[bag_param])

    def test_sm(self):
        self.assertAlmostEqual(flavio.sm_prediction('tau_B+/tau_Bd'), 1.08381512025)

    def test_WE_cu(self):
        self.assertEqual(flavio.physics.bdecays.lifetime_ratio.weak_exchange(wc_sm, par, "B0"), 0)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLL_bcud": 1}, scale=4.5)
        self.assertAlmostEqual(1e15*flavio.physics.bdecays.lifetime_ratio.weak_exchange(wc, par, "B0"), 3.03733567, places=6)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLLt_bcud": 1}, scale=4.5)
        self.assertAlmostEqual(1e15*flavio.physics.bdecays.lifetime_ratio.weak_exchange(wc, par, "B0"), 9.79346592, places=5)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLLt_bcus": 1}, scale=4.5)
        self.assertEqual(flavio.physics.bdecays.lifetime_ratio.weak_exchange(wc, par, "B0"), 0)

    def test_PI_cd(self):
        self.assertEqual(flavio.physics.bdecays.lifetime_ratio.pauli_interference(wc_sm, par, "B+"), 0)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLL_bcud": 1}, scale=4.5)
        self.assertAlmostEqual(1e14*flavio.physics.bdecays.lifetime_ratio.pauli_interference(wc, par, "B+"), 2.82494420, places=6)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLL_bcus": 1}, scale=4.5)
        self.assertAlmostEqual(1e14*flavio.physics.bdecays.lifetime_ratio.pauli_interference(wc, par, "B+"), 0.1503590699, places=6)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLLt_bcud": 1}, scale=4.5)
        self.assertAlmostEqual(1e13*flavio.physics.bdecays.lifetime_ratio.pauli_interference(wc, par, "B+"), 1.38771099, places=6)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLLt_bcus": 1}, scale=4.5)
        Vud = flavio.physics.ckm.get_ckm(par)[0,0]
        Vus = flavio.physics.ckm.get_ckm(par)[0,1]
        self.assertAlmostEqual(1e13*flavio.physics.bdecays.lifetime_ratio.pauli_interference(wc, par, "B+"), 1.38771099*(Vus/Vud)**2, places=6)

    def test_NP_bcud(self):
        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLL_bcud": -2}, scale=4.5)
        self.assertAlmostEqual(flavio.np_prediction("tau_B+/tau_Bd", wc), 1.011517673)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLLt_bcud": 3}, scale=4.5)
        self.assertAlmostEqual(flavio.np_prediction("tau_B+/tau_Bd", wc), 0.007740969, places=6)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVRLt_bcud": -0.15, "CSLL_bcud": 0.2}, scale=4.5)
        self.assertAlmostEqual(flavio.np_prediction("tau_B+/tau_Bd", wc), 1.113086143)

    def test_NP_bcus(self):
        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLL_bcus": -2}, scale=4.5)
        self.assertAlmostEqual(flavio.np_prediction("tau_B+/tau_Bd", wc), 1.079837520)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLLt_bcus": 3}, scale=4.5)
        self.assertAlmostEqual(flavio.np_prediction("tau_B+/tau_Bd", wc), 1.017205203)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVRLt_bcus": -0.15, "CSLL_bcus": 0.2}, scale=4.5)
        self.assertAlmostEqual(flavio.np_prediction("tau_B+/tau_Bd", wc), 1.083496791)

    def test_NP_dbcc(self):
        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLL_bdcc": -2}, scale=4.5)
        self.assertAlmostEqual(flavio.np_prediction("tau_B+/tau_Bd", wc), 1.084098486)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLLt_bdcc": 3}, scale=4.5)
        self.assertAlmostEqual(flavio.np_prediction("tau_B+/tau_Bd", wc), 1.102675260)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVRLt_bdcc": -0.15, "CSLL_bdcc": 0.2}, scale=4.5)
        self.assertAlmostEqual(flavio.np_prediction("tau_B+/tau_Bd", wc), 1.085435224)
