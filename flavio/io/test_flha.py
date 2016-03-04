import unittest
import numpy as np
from . import flha
import flavio
import pkgutil
import os

class TestFLHA(unittest.TestCase):
    def test_flha(self):
        test_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'test', 'SPheno.spc.MSSM')
        wc = flha.read_wilson(test_file)
        self.assertIsInstance(wc, flavio.WilsonCoefficients)
