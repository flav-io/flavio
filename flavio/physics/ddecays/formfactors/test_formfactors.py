import unittest
import flavio
from flavio.classes import AuxiliaryQuantity

constraints = flavio.default_parameters
wc_obj = flavio.WilsonCoefficients()
par = constraints.get_central_all()


class TestDPFF(unittest.TestCase):
    def test_etm(self):
        # compare to ETM result at kinematic endpoints
        q2max_pi = (par['m_D0'] - par['m_pi+'])**2
        q2max_K = (par['m_D0'] - par['m_K+'])**2
        ff_pi_0 = AuxiliaryQuantity["D->pi form factor"].prediction_central(constraints, wc_obj, 0)
        ff_K_0 = AuxiliaryQuantity["D->K form factor"].prediction_central(constraints, wc_obj, 0)
        ff_pi_max = AuxiliaryQuantity["D->pi form factor"].prediction_central(constraints, wc_obj, q2max_pi)
        ff_K_max = AuxiliaryQuantity["D->K form factor"].prediction_central(constraints, wc_obj, q2max_K)
        self.assertAlmostEqual(ff_pi_0['f0'], 0.612, delta=0.001)
        self.assertAlmostEqual(ff_pi_max['f0'], 1.134, delta=0.01)
        self.assertAlmostEqual(ff_pi_0['f+'], 0.612, delta=0.001)
        self.assertAlmostEqual(ff_pi_max['f+'], 2.130, delta=0.02)
        self.assertAlmostEqual(ff_pi_0['fT'], 0.506, delta=0.03)
        self.assertAlmostEqual(ff_pi_max['fT'], 1.573, delta=0.03)
        self.assertAlmostEqual(ff_K_0['f0'], 0.765, delta=0.001)
        self.assertAlmostEqual(ff_K_max['f0'], 0.979, delta=0.001)
        self.assertAlmostEqual(ff_K_0['f+'], 0.765, delta=0.001)
        self.assertAlmostEqual(ff_K_max['f+'], 1.336, delta=0.01)
        self.assertAlmostEqual(ff_K_0['fT'], 0.687, delta=0.001)
        self.assertAlmostEqual(ff_K_max['fT'], 1.170, delta=0.01)
