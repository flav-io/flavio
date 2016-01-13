import unittest
from flavio.physics.pdg import read_pdg_masswidth
import numpy as np

# 1 second
s = 1.519267515435317e+24
ps = 1e-12*s

class TestPDG(unittest.TestCase):
    particles = read_pdg_masswidth('data/pdg/mass_width_2015.mcd')
    def test_pdg(self):
        # check top mass central value
        self.assertEqual(self.particles['t+2/3']['mass'][0], 173.21)
        # compare B_s lifetime and errors in picoseconds to inverse width
        GammaBs = self.particles['B(s)0']['width']
        self.assertAlmostEqual(1/GammaBs[0]/ps, 1.510, places=3)
        self.assertAlmostEqual(1/ps*GammaBs[1]/GammaBs[0]**2, +0.005, places=3)
        self.assertAlmostEqual(1/ps*GammaBs[2]/GammaBs[0]**2, -0.005, places=3)
