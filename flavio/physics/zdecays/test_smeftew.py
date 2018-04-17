import unittest
import flavio
from .smeftew import *


class ZeroDict(dict):
    """Dictionary that always returns 0"""

    def __init__(self, arg):
        super().__init__()

    def __getitem__(self, arg):
        return 0


class TestSMEFTew(unittest.TestCase):
    def test_shifts_sm(self):
        C = ZeroDict({})
        par = flavio.default_parameters.get_central_all()
        self.assertEqual(d_GF(par, C), 0)
        self.assertEqual(d_mZ2(par, C), 0)
        self.assertEqual(d_gZb(par, C), 0)
