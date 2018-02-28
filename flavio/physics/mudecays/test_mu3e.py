import unittest
import flavio


class TestMu3E(unittest.TestCase):
    def test_mu3e(self):
        self.assertEqual(flavio.sm_prediction('BR(mu->eee)'), 0)
