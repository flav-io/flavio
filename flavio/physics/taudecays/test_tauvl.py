import unittest
import flavio


class TestTauVl(unittest.TestCase):
    def test_tauvl(self):
        # compare to the experimental values
        self.assertEqual(flavio.sm_prediction('BR(tau->rhoe)'), 0)
        self.assertEqual(flavio.sm_prediction('BR(tau->rhomu)'), 0)
