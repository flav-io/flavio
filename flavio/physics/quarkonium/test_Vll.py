import unittest
import flavio


par = flavio.default_parameters.get_central_all()

### implement test
class TestVll(unittest.TestCase):
    def test_sm(self):
        self.assertEqual(flavio.sm_prediction('BR(J/psi->ee)'), 1.) #par['tau_Bc_SM'])
#        self.assertAlmostEqual(flavio.sm_prediction('tau_Bc'), par['tau_Bc'], delta=0.1 * par['tau_Bc'])
