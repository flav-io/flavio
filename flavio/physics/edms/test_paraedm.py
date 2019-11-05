import unittest
import flavio
from wilson import wcxf
from math import sqrt


par = flavio.default_parameters.get_central_all()


class TestParamagneticEDM(unittest.TestCase):
    def test_sm(self):
        self.assertEqual(flavio.sm_prediction('d_Tl'), 0)
        self.assertEqual(flavio.sm_prediction('omega_YbF'), 0)
        self.assertEqual(flavio.sm_prediction('omega_HfF'), 0)
        self.assertEqual(flavio.sm_prediction('omega_ThO'), 0)
