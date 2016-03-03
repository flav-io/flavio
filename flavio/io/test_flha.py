import unittest
import numpy as np
from . import flha
import flavio
import pkgutil
import os

class TestFLHA(unittest.TestCase):
    def test_flha(self):
        test_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'test', 'SPheno.spc.MSSM')
        print(test_file)
        wc = flha.get_wc_from_file(test_file)
        self.assertIsInstance(wc, flavio.WilsonCoefficients)
        print(wc.get_wc('bsmumu', 160, flavio.default_parameters.get_central_all()))
