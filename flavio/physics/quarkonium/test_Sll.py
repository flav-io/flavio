import unittest
import flavio
from wilson import Wilson
from .Sll import *

### implement test
class TestSll(unittest.TestCase):
    def test_np(self):
        wc,br=Wilson({'CSRR_muecc' : 1e-2},scale=2.,eft='WET',basis='flavio'),0.0014399
        self.assertAlmostEqual(flavio.np_prediction('BR(chi_c0(1P)->mue)',wc), br,delta=0.001*br)

        wc,br=Wilson({'CSRR_tauecc' : 1e-2},scale=2.,eft='WET',basis='flavio'),0.00076716
        self.assertAlmostEqual(flavio.np_prediction('BR(chi_c0(1P)->taue)',wc), br,delta=0.001*br)
