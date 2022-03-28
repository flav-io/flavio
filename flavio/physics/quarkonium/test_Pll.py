import unittest
import flavio
from wilson import Wilson
from .Pll import *

### implement test
class TestPll(unittest.TestCase):
    def test_np(self):
        wc,br=Wilson({'CVRR_muecc' : 1e-2},scale=2.,eft='WET',basis='flavio'),7.7348e-8
        self.assertAlmostEqual(flavio.np_prediction('BR(eta_c->mue)',wc), br,delta=0.01*br)

        wc,br=Wilson({'CSRR_muecc' : 1e-2},scale=2.,eft='WET',basis='flavio'),8.5135e-5
        self.assertAlmostEqual(flavio.np_prediction('BR(eta_c->mue)',wc), br,delta=0.01*br)

        wc,br=Wilson({'CVRR_tauecc' : 1e-2},scale=2.,eft='WET',basis='flavio'),9.1346e-6
        self.assertAlmostEqual(flavio.np_prediction('BR(eta_c->taue)',wc), br,delta=0.01*br)
        
        wc,br=Wilson({'CSRR_tauecc' : 1e-2},scale=2.,eft='WET',basis='flavio'),3.5551e-5
        self.assertAlmostEqual(flavio.np_prediction('BR(eta_c->taue)',wc), br,delta=0.01*br)
