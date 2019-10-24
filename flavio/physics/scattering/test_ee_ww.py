import unittest
from flavio import sm_prediction, np_prediction
import wilson
import numpy as np


Es = np.array([161.3, 172.1, 182.7, 188.6, 191.6, 195.5, 199.5, 201.6, 204.9, 206.6])


class TestEEWW(unittest.TestCase):

    def test_ee_ww_SM(self):
        for E in Es:
            self.assertEqual(sm_prediction('R(ee->WW)', E=E), 1)

    def test_ee_ww_GF(self):
        """Check that the NP contribution from phil3_22 and ll_1221
        differs by a factor -2, since they only enter through the modified
        Fermi constant."""
        np1 = wilson.Wilson({'phil3_22': 1e-6}, 91.1876, 'SMEFT', 'Warsaw')
        np2 = wilson.Wilson({'ll_1221': 1e-6}, 91.1876, 'SMEFT', 'Warsaw')
        for E in Es:
            R1 = np_prediction('R(ee->WW)', E=E, wc_obj=np1)
            R2 = np_prediction('R(ee->WW)', E=E, wc_obj=np2)
            self.assertAlmostEqual((R1 - 1) / (R2 - 1), -2,
                                   delta=0.2,
                                   msg="Failed for {}".format(E))

    def test_diff_ee_ww_NP(self):
        coeffs = ['phiWB', 'phiD', 'phil3_11', 'phil3_22', 'll_1221', 'phil1_11', 'phie_11']
        for coeff in coeffs:
            for E in [182.66, 189.09, 198.38, 205.92]:
                _E = Es.flat[np.abs(Es - E).argmin()]
                dsigma = []
                dsigma_sm = []
                for i in range(10):
                    args = (E,
                            np.round(i * 0.2 - 1, 1),
                            np.round((i + 1)  * 0.2 - 1, 1))
                    w = wilson.Wilson({coeff: 0.1 / 246.22**2}, 91.1876, 'SMEFT', 'Warsaw')
                    dsigma.append(np_prediction('<dR/dtheta>(ee->WW)', w, *args))
                    dsigma_sm.append(sm_prediction('<dR/dtheta>(ee->WW)', *args))
                r_tot = np_prediction('R(ee->WW)', w, _E)
                sigma_tot_sm = sum(dsigma_sm)
                sigma_tot = sum(dsigma)
                self.assertAlmostEqual(sigma_tot / sigma_tot_sm,
                                       r_tot,
                                       delta=0.25,
                                       msg="Failed for E={}, C_{}".format(E, coeff))
