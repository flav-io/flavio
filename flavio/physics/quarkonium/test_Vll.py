import unittest
import flavio


par = flavio.default_parameters.get_central_all()

### implement test
class TestVll(unittest.TestCase):
    def test_sm(self):
        br=par['BR(J/psi->e+e-)']
        self.assertEqual(flavio.sm_prediction('BR(J/psi->ee)'),br ) #par['tau_Bc_SM'])
        self.assertAlmostEqual(flavio.sm_prediction('BR(J/psi->ee)'), br,delta=0.1*br)
