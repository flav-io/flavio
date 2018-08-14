import unittest
import flavio


class TestEpsPrime(unittest.TestCase):
    def test_epspsm(self):
        # check the SM prediction is in the right ball park
        self.assertAlmostEqual(1e4 * flavio.sm_prediction('epsp/eps'),
                               0.5,
                               delta=1)
