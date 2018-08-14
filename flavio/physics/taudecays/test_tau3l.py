import unittest
import flavio


class TestTau3l(unittest.TestCase):
    def test_tau3l(self):
        # compare to the experimental values
        self.assertEqual(flavio.sm_prediction('BR(tau->muee)'), 0)
        self.assertEqual(flavio.sm_prediction('BR(tau->mumumu)'), 0)
