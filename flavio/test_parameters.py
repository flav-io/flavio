import unittest
import numpy as np
from .parameters import *
from .classes import *
from .parameters import FlavioParticle, p_data
import tempfile

s = 1.519267515435317e+24
ps = 1e-12*s


class TestPDG(unittest.TestCase):
    year = 2018
    FlavioParticle.load_table(p_data.open_text(p_data, "particle{}.csv".format(year)))
    def test_pdg(self):
        # check some tex names and masses
        to_check = {
            'Bs': ('B_{s}', 5.366890000000001),
            'Bc': ('B_{c}', 6.2749),
            'Bs*': ('B_{s}^{*}', 5.4154),
            'B*+': ('B^{*+}', 5.32465),
            'B*0': ('B^{*0}', 5.32465),
            'B+': ('B^{+}', 5.27932),
            'B0': ('B^{0}', 5.27963),
            'Ds': ('D_{s}', 1.96834),
            'Ds*': ('D_{s}^{*}', 2.1122),
            'D+': ('D^{+}', 1.86965),
            'D0': ('D^{0}', 1.86483),
            'h': ('H', 125.18),
            'J/psi': ('J/\\psi', 3.0969),
            'KL': ('K_{L}', 0.497611),
            'KS': ('K_{S}', 0.497611),
            'K*+': ('K^{*+}', 0.89176),
            'K*0': ('K^{*0}', 0.89555),
            'K+': ('K^{+}', 0.49367700000000003),
            'K0': ('K^{0}', 0.497611),
            'Lambda': ('\\Lambda', 1.115683),
            'Lambdab': ('\\Lambda_{b}', 5.6196),
            'Lambdac': ('\\Lambda_{c}', 2.28646),
            'omega': ('\\omega', 0.78265),
            'D*0': ('D^{*0}', 2.00685),
            'D*+': ('D^{*+}', 2.01026),
            'W': ('W', 80.379),
            'Z': ('Z', 91.1876),
            'e': ('e', 0.0005109989461),
            'eta': ('\\eta', 0.547862),
            'f0': ('f_{0}', 0.99),
            'mu': ('\\mu', 0.1056583745),
            'phi': ('\\phi', 1.019461),
            'pi+': ('\\pi^{+}', 0.13957060999999998),
            'pi0': ('\\pi^{0}', 0.134977),
            'psi(2S)': ('\\psi_{2S}', 3.686097),
            'rho+': ('\\rho^{+}', 0.7752600000000001),
            'rho0': ('\\rho^{0}', 0.7752600000000001),
            't': ('t', 173.1),
            'tau': ('\\tau', 1.7768599999999999),
            'u': ('u', 0.0022),
            'p': ('p', 0.9382720809999999),
            'n': ('n', 0.9395654130000001),
        }
        for flavio_name, (tex_test, mass_test) in to_check.items():
            particle = FlavioParticle.from_flavio_name(flavio_name)
            self.assertEqual(particle.latex_name_simplified, tex_test)
            self.assertEqual(particle.flavio_m[3], mass_test)
        # check B_s lifetime and errors in picoseconds
        particle = FlavioParticle.from_flavio_name('Bs')
        tauBs = particle.flavio_tau[3:]
        self.assertAlmostEqual(tauBs[0]/ps, 1.509, places=3)
        self.assertAlmostEqual(tauBs[1]/ps, 0.004, places=3)
        self.assertAlmostEqual(tauBs[2]/ps, 0.004, places=3)

class TestParameters(unittest.TestCase):
    def test_parameters(self):
        par_dict = default_parameters.get_central_all()
        # parameters from the YAML file
        self.assertEqual(par_dict['alpha_s'],  0.1182)
        self.assertEqual(par_dict['Gamma12_Bs_c'],  -48.0)
        # parameters from the PDG file
        self.assertEqual(par_dict['m_W'], 80.379)
        self.assertEqual(par_dict['tau_phi'], 1/4.249e-3)
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
