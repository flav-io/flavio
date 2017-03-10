import unittest
import numpy as np
from .measurements import *
from .classes import *
import tempfile
from .measurements import _load_new


class TestMeasurements(unittest.TestCase):
    def test_measurements(self):
        m = Measurement['Belle phigamma 2014']
        self.assertEqual(m.experiment, 'Belle')

    def test_yaml_io_new(self):
        # read_file for a single measurement
        m1 = Measurement['Belle phigamma 2014']
        with tempfile.NamedTemporaryFile('r+') as tf:
            tf.write(m1.get_yaml(pname='observables'))
            tf.seek(0) # rewind
            m2 = read_file(tf.name)
            m2 = [Measurement[m] for m in m2]
        self.assertEqual(m1.get_yaml_dict(), m2[0].get_yaml_dict())
        # and now for 2 measurements
        m1 = [Measurement['Belle phigamma 2014'], Measurement['HFAG rad 2014']]
        with tempfile.NamedTemporaryFile('r+') as tf:
            write_file(tf.name, m1)
            tf.seek(0) # rewind
            m2 = read_file(tf.name)
            m2 = [Measurement[m] for m in m2]
        for i in range(2):
            self.assertEqual(m1[i].get_yaml_dict(), m2[i].get_yaml_dict())
        # and again but using the string names in write_file
        with tempfile.NamedTemporaryFile('r+') as tf:
            write_file(tf.name, [m.name for m in m1])
            tf.seek(0) # rewind
            m2 = read_file(tf.name)
            m2 = [Measurement[m] for m in m2]
        for i in range(2):
            self.assertEqual(m1[i].get_yaml_dict(), m2[i].get_yaml_dict())
