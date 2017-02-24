import unittest
import numpy as np
from .measurements import *
from .classes import *


class TestMeasurements(unittest.TestCase):
    def test_measurements(self):
        m = Measurement['Belle phigamma 2014']
        self.assertEqual(m.experiment, 'Belle')
