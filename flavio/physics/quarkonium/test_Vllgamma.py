import unittest
import flavio
from wilson import Wilson
from .Vllgamma import *

### implement test
class TestVllgamma(unittest.TestCase):
    def test_np(self):
        wc,br=Wilson({'CVRR_muecc' : 1e-2},scale=2.,eft='WET',basis='flavio'),7.7736e-10
        self.assertAlmostEqual(flavio.np_prediction('BR(J/psi->muegamma)',wc), br,delta=0.001*br)

        wc,br=Wilson({'CSRR_muecc' : 1e-2},scale=2.,eft='WET',basis='flavio'),5.8277e-10
        self.assertAlmostEqual(flavio.np_prediction('BR(J/psi->muegamma)',wc), br,delta=0.001*br)

        wc,br=Wilson({'CVRR_tauecc' : 1e-2},scale=2.,eft='WET',basis='flavio'),1.1934e-10
        self.assertAlmostEqual(flavio.np_prediction('BR(J/psi->tauegamma)',wc), br,delta=0.001*br)
        
        wc,br=Wilson({'CSRR_tauecc' : 1e-2},scale=2.,eft='WET',basis='flavio'),8.4356e-11
        self.assertAlmostEqual(flavio.np_prediction('BR(J/psi->tauegamma)',wc), br,delta=0.001*br)
