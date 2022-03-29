import unittest
import flavio
from wilson import Wilson
from .Vll import *

par = flavio.default_parameters.get_central_all()

### implement test
class TestVll(unittest.TestCase):
    def test_sm(self):
        par=flavio.default_parameters.get_central_all()
        br=par['Gamma_SM(J/psi->ee)']/par['Gamma_J/psi']
#        self.assertEqual(flavio.sm_prediction('BR(J/psi->ee)'),br ) 
        self.assertAlmostEqual(flavio.sm_prediction('BR(J/psi->ee)'), br,delta=0.1*br)

    def test_np(self):
        wc,br=Wilson({'CVRR_muecc' : 1e-2},scale=2.,eft='WET',basis='flavio'),0.01788
#        self.assertEqual(flavio.np_prediction('BR(J/psi->mue)',wc),br ) 
        self.assertAlmostEqual(flavio.np_prediction('BR(J/psi->mue)',wc), br,delta=0.001*br)
        self.assertAlmostEqual(flavio.np_prediction('R(J/psi->mue)',wc), flavio.np_prediction('BR(J/psi->mue)',wc)/flavio.np_prediction('BR(J/psi->ee)',wc),delta=0.001*br)

        wc,br=Wilson({'CTRR_muecc' : 1e-2},scale=2.,eft='WET',basis='flavio'),0.1312
#        self.assertEqual(flavio.np_prediction('BR(J/psi->mue)',wc),br ) 
        self.assertAlmostEqual(flavio.np_prediction('BR(J/psi->mue)',wc), br,delta=0.001*br)
        self.assertAlmostEqual(flavio.np_prediction('R(J/psi->mue)',wc), flavio.np_prediction('BR(J/psi->mue)',wc)/flavio.np_prediction('BR(J/psi->ee)',wc),delta=0.001*br)

        wc,br=Wilson({'CVRR_tauecc' : 1e-2},scale=2.,eft='WET',basis='flavio'),0.009387
#        self.assertEqual(flavio.np_prediction('BR(J/psi->taue)',wc),br ) 
        self.assertAlmostEqual(flavio.np_prediction('BR(J/psi->taue)',wc), br,delta=0.001*br)
        self.assertAlmostEqual(flavio.np_prediction('R(J/psi->taue)',wc), flavio.np_prediction('BR(J/psi->taue)',wc)/flavio.np_prediction('BR(J/psi->ee)',wc),delta=0.001*br)
        
        wc,br=Wilson({'CTRR_tauecc' : 1e-2},scale=2.,eft='WET',basis='flavio'),0.09791
#        self.assertEqual(flavio.np_prediction('BR(J/psi->taue)',wc),br ) 
        self.assertAlmostEqual(flavio.np_prediction('BR(J/psi->taue)',wc), br,delta=0.001*br)
        self.assertAlmostEqual(flavio.np_prediction('R(J/psi->taue)',wc), flavio.np_prediction('BR(J/psi->taue)',wc)/flavio.np_prediction('BR(J/psi->ee)',wc),delta=0.001*br)
