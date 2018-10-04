import unittest
import numpy as np
import numpy.testing as npt
from .measurements import *
from .classes import *
import tempfile
from .measurements import _load_new, _fix_correlation_matrix


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

    def test_fix_correlation(self):
        npt.assert_array_equal(
            _fix_correlation_matrix(0.3, 2),
            np.array([[1, 0.3], [0.3, 1]]))
        npt.assert_array_equal(
            _fix_correlation_matrix(0.3, 3),
            np.array([[1, 0.3, 0.3], [0.3, 1, 0.3], [0.3, 0.3, 1]]))
        npt.assert_array_equal(
            _fix_correlation_matrix([[1, 0.4, 0.3], [1, 0.2], [1]], 3),
            np.array([[1, 0.4, 0.3], [0.4, 1, 0.2], [0.3, 0.2, 1]]))

    def test_measurements_yaml(self):
        # check if all observables in existing measurements exist
        for name, m in flavio.Measurement.instances.items():
            for obs in m.all_parameters:
                if 'test' in obs or 'test' in name:
                    continue  # ignore observables defined in unit tests
                # this will raise if the observable does not exist
                obsname = flavio.Observable.argument_format(obs, 'dict')['name']
                flavio.Observable[obsname]
