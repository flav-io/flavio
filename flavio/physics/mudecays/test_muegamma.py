import unittest
import flavio


class TestMuEGamma(unittest.TestCase):
    def test_muegamma(self):
        self.assertEqual(flavio.sm_prediction('BR(mu->egamma)'), 0)
