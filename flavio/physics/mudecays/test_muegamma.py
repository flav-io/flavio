import unittest
import flavio


class TestTauLGamma(unittest.TestCase):
    def test_taulgamma(self):
        # compare to the experimental values
        self.assertEqual(flavio.sm_prediction('BR(mu->egamma)'), 0)
