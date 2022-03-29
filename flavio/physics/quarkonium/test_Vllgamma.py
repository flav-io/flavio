import unittest
import flavio
from wilson import Wilson
from .Vllgamma import *

### implement test
class TestVllgamma(unittest.TestCase):
    def test_np(self):
        wc,br=Wilson({'CVRR_muecc' : 1e-2},scale=2.,eft='WET',basis='flavio'),8.3949e-6
        self.assertAlmostEqual(flavio.np_prediction('BR(J/psi->muegamma)',wc), br,delta=0.01*br)
        self.assertAlmostEqual(flavio.np_prediction('R(J/psi->muegamma)',wc),flavio.np_prediction('BR(J/psi->muegamma)',wc)/flavio.np_prediction('BR(J/psi->ee)',wc),delta=0.001*br)

        wc,br=Wilson({'CSRR_muecc' : 1e-2},scale=2.,eft='WET',basis='flavio'),6.2935e-6
        self.assertAlmostEqual(flavio.np_prediction('BR(J/psi->muegamma)',wc), br,delta=0.01*br)
        self.assertAlmostEqual(flavio.np_prediction('R(J/psi->muegamma)',wc),flavio.np_prediction('BR(J/psi->muegamma)',wc)/flavio.np_prediction('BR(J/psi->ee)',wc),delta=0.001*br)

        wc,br=Wilson({'CVRR_tauecc' : 1e-2},scale=2.,eft='WET',basis='flavio'),1.2887e-6
        self.assertAlmostEqual(flavio.np_prediction('BR(J/psi->tauegamma)',wc), br,delta=0.01*br)
        self.assertAlmostEqual(flavio.np_prediction('R(J/psi->tauegamma)',wc),flavio.np_prediction('BR(J/psi->tauegamma)',wc)/flavio.np_prediction('BR(J/psi->ee)',wc),delta=0.001*br)
        
        wc,br=Wilson({'CSRR_tauecc' : 1e-2},scale=2.,eft='WET',basis='flavio'),9.1097e-7
        self.assertAlmostEqual(flavio.np_prediction('BR(J/psi->tauegamma)',wc), br,delta=0.01*br)
        self.assertAlmostEqual(flavio.np_prediction('R(J/psi->tauegamma)',wc),flavio.np_prediction('BR(J/psi->tauegamma)',wc)/flavio.np_prediction('BR(J/psi->ee)',wc),delta=0.001*br)

        