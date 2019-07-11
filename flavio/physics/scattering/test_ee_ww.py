import unittest
import flavio
import wilson


Es = [161.3, 172.1, 182.7, 188.6, 191.6, 195.5, 199.5, 201.6, 204.9, 206.6]


class TestEEWW(unittest.TestCase):

    def test_ee_ww_SM(self):
        for E in Es:
            self.assertEqual(flavio.sm_prediction('R(ee->WW)', E=E), 1)

    def test_ee_ww_GF(self):
        """Check that the NP contribution from phil3_22 and ll_1221
        differs by a factor -2, since they only enter through the modified
        Fermi constant."""
        np1 = wilson.Wilson({'phil3_22': 1e-6}, 91.1876, 'SMEFT', 'Warsaw')
        np2 = wilson.Wilson({'ll_1221': 1e-6}, 91.1876, 'SMEFT', 'Warsaw')
        for E in Es:
            R1 = flavio.np_prediction('R(ee->WW)', E=E, wc_obj=np1)
            R2 = flavio.np_prediction('R(ee->WW)', E=E, wc_obj=np2)
            self.assertAlmostEqual((R1 - 1) / (R2 - 1), -2,
                                   delta=0.02,
                                   msg="Failed for {}".format(E))
