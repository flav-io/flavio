import unittest
import numpy as np
from .parameters import *
from .classes import *
from .parameters import _read_pdg_masswidth
import tempfile

s = 1.519267515435317e+24
ps = 1e-12*s


class TestPDG(unittest.TestCase):
    particles = _read_pdg_masswidth('data/pdg/mass_width_2015.mcd')
    def test_pdg(self):
        # check some masses
        self.assertEqual(self.particles['K*(892)+']['mass'][0], 0.89166)
        self.assertEqual(self.particles['K*(892)0']['mass'][0], 0.89581)
        self.assertEqual(self.particles['t']['mass'][0], 173.21)
        # compare B_s lifetime and errors in picoseconds to inverse width
        GammaBs = self.particles['B(s)']['width']
        self.assertAlmostEqual(1/GammaBs[0]/ps, 1.510, places=3)
        self.assertAlmostEqual(1/ps*GammaBs[1]/GammaBs[0]**2, +0.005, places=3)
        self.assertAlmostEqual(1/ps*GammaBs[2]/GammaBs[0]**2, -0.005, places=3)

class TestParameters(unittest.TestCase):
    def test_parameters(self):
        par_dict = default_parameters.get_central_all()
        # parameters from the YAML file
        self.assertEqual(par_dict['alpha_s'],  0.1185)
        self.assertEqual(par_dict['Gamma12_Bs_c'],  -48.0)
        # parameters from the PDG file
        self.assertEqual(par_dict['m_W'], 80.385)
        self.assertEqual(par_dict['tau_phi'], 1/4.247e-3)
        # just check if the random values are numbers
        for par_random in default_parameters.get_random_all().values():
            self.assertIsInstance(par_random, float)
        for par_random in default_parameters.get_random_all(size=2).values():
            self.assertEqual(par_random.shape, (2,))

    def test_constraints_from_string(self):
        pds = constraints_from_string('1.36(34)(3) 1e-3')
        for pd in pds:
            self.assertEqual(pd.get_central(), 1.36e-3)
        self.assertEqual(pds[0].standard_deviation, 0.34e-3)
        self.assertEqual(pds[1].standard_deviation, 0.03e-3)

        pds = constraints_from_string('1.36(1.4)(3)1E5')
        for pd in pds:
            self.assertEqual(pd.get_central(), 1.36e5)
        self.assertEqual(pds[0].standard_deviation, 1.4e5)
        self.assertEqual(pds[1].standard_deviation, 0.03e5)

        pds = constraints_from_string('5(3)(2)')
        for pd in pds:
            self.assertEqual(pd.get_central(), 5.)
        self.assertEqual(pds[0].standard_deviation, 3.)
        self.assertEqual(pds[1].standard_deviation, 2.)

        pds = constraints_from_string('1.36 +- 0.34 +- 0.3 * 10^-5')
        for pd in pds:
            self.assertAlmostEqual(pd.get_central(), 1.36e-5, places=12)
        self.assertAlmostEqual(pds[0].standard_deviation, 0.34e-5, places=12)
        self.assertAlmostEqual(pds[1].standard_deviation, 0.3e-5, places=12)

        pds = constraints_from_string('1.36±0.34±0.3')
        for pd in pds:
            self.assertEqual(pd.get_central(), 1.36)
        self.assertEqual(pds[0].standard_deviation, 0.34)
        self.assertEqual(pds[1].standard_deviation, 0.3)

        pds = constraints_from_string(r'1.36 \pm 0.34 \pm 0.3')
        for pd in pds:
            self.assertEqual(pd.get_central(), 1.36)
        self.assertEqual(pds[0].standard_deviation, 0.34)
        self.assertEqual(pds[1].standard_deviation, 0.3)

        pds = constraints_from_string('5 + 0.1 - 0.2 + 0.3 - 0.5')
        for pd in pds:
            self.assertEqual(pd.get_central(), 5.)
        self.assertEqual(pds[0].left_deviation, 0.2)
        self.assertEqual(pds[0].right_deviation, 0.1)
        self.assertEqual(pds[1].left_deviation, 0.5)
        self.assertEqual(pds[1].right_deviation, 0.3)

    def test_yaml_io_new(self):
        from flavio import default_parameters
        with tempfile.NamedTemporaryFile('r+') as tf:
            write_file(tf.name, default_parameters)
            tf.seek(0) # rewind
            c = read_file(tf.name)
        self.assertEqual(c.get_yaml_dict(), default_parameters.get_yaml_dict())
