import unittest
import flavio


class TestTau3l(unittest.TestCase):
    def test_tau3l_sm(self):
        self.assertEqual(flavio.sm_prediction('BR(tau->muee)'), 0)
        self.assertEqual(flavio.sm_prediction('BR(tau->mumumu)'), 0)
        self.assertEqual(flavio.sm_prediction('BR(tau->emumu)'), 0)
        self.assertEqual(flavio.sm_prediction('BR(tau->eee)'), 0)
        self.assertEqual(flavio.sm_prediction('BR(tau->emue)'), 0)
        self.assertEqual(flavio.sm_prediction('BR(tau->muemu)'), 0)
