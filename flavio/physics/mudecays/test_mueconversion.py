import unittest
import flavio

class TestMuEConversion(unittest.TestCase):
	def test_mueconversion(self):
		self.assertEqual(flavio.sm_prediction('CR(mu->e, Au)'), 0)
		self.assertEqual(flavio.sm_prediction('CR(mu->e, Al)'), 0)
		self.assertEqual(flavio.sm_prediction('CR(mu->e, Ti)'), 0)