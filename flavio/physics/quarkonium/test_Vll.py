import unittest
import flavio
from wilson import Wilson
from .Vll import *

par = flavio.default_parameters.get_central_all()

### implement test
class TestVll(unittest.TestCase):
    def test_sm(self):
        br=0.05971 # PDG 2021 experimental value 
        self.assertEqual(flavio.sm_prediction('BR(J/psi->ee)'),br ) 
        self.assertAlmostEqual(flavio.sm_prediction('BR(J/psi->ee)'), br,delta=0.1*br)

    def test_np(self):
        wc,br=Wilson({'CVRR_muecc' : 1e-2},scale=2.,eft='WET',basis='flavio'),0.01855
        self.assertEqual(flavio.np_prediction('BR(J/psi->mue)',wc),br ) 
        self.assertAlmostEqual(flavio.sm_prediction('BR(J/psi->mue)',wc), br,delta=0.1*br)

        wc,br=Wilson({'CTRR_muecc' : 1e-2},scale=2.,eft='WET',basis='flavio'),0.143023
        self.assertEqual(flavio.np_prediction('BR(J/psi->mue)',wc),br ) 
        self.assertAlmostEqual(flavio.sm_prediction('BR(J/psi->mue)',wc), br,delta=0.1*br)

        wc,br=Wilson({'CVRR_tauecc' : 1e-2},scale=2.,eft='WET',basis='flavio'),0.009738
        self.assertEqual(flavio.np_prediction('BR(J/psi->mue)',wc),br ) 
        self.assertAlmostEqual(flavio.sm_prediction('BR(J/psi->mue)',wc), br,delta=0.1*br)
        
        wc,br=Wilson({'CTRR_tauecc' : 1e-2},scale=2.,eft='WET',basis='flavio'),0.10673
        self.assertEqual(flavio.np_prediction('BR(J/psi->mue)',wc),br ) 
        self.assertAlmostEqual(flavio.sm_prediction('BR(J/psi->mue)',wc), br,delta=0.1*br)
