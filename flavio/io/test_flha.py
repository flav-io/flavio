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

    def test_ckm(self):
        par = flavio.default_parameters.copy()

        test_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'test', 'SPheno.spc.MSSM')
        with self.assertRaisesRegexp(ValueError, ".*does not contain a VCKMIN block.*"):
            flha.read_ckm(test_file, par)

        test_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'test', 'SPheno-2.spc.MSSM')
        with self.assertLogs('SLHA', level='WARN') as cm:
            flha.read_ckm(test_file, par)
            self.assertEqual(cm.output, ['WARNING:SLHA:CKM matrix parametrization is not set to "Wolfenstein". read_ckm will have no effect!'])
        self.assertEqual(par.get_central('laC'), 2.25649637E-01)
        self.assertEqual(par.get_central('A'), 8.04207424E-01)
        self.assertEqual(par.get_central('rhobar'), 1.94560639E-01)
        self.assertEqual(par.get_central('etabar'), 4.55377988E-01)
