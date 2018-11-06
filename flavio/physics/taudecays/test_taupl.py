import unittest
import flavio

class TestTauPl(unittest.TestCase):
    def test_taupl(self):
        # check SM prediction
        self.assertEqual(flavio.sm_prediction('BR(tau->pie)'), 0)
        self.assertEqual(flavio.sm_prediction('BR(tau->pimu)'), 0)
        self.assertEqual(flavio.sm_prediction('BR(tau->Ke)'), 0)
        self.assertEqual(flavio.sm_prediction('BR(tau->Kmu)'), 0)
