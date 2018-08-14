import unittest
import flavio


class TestTauLGamma(unittest.TestCase):
    def test_taulgamma(self):
        # compare to the experimental values
        self.assertEqual(flavio.sm_prediction('BR(tau->mugamma)'), 0)
        self.assertEqual(flavio.sm_prediction('BR(tau->egamma)'), 0)
