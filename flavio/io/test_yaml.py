import unittest
import flavio
import os

class TestYAML(unittest.TestCase):
    def test_load_include(self):
        test_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'test', '1.yaml')
        with open(test_file, 'r') as f:
            d = flavio.io.yaml.load_include(f)
        self.assertDictEqual(d, {'key': {'key2': 0},
                                 'key3': [1, 2, 3, 4, 5, 6, 7, 8]})
    def test_load_include_error(self):
        test_str = "key: !include relative_path.yaml"
        with self.assertRaises(ValueError):
            d = flavio.io.yaml.load_include(test_str)
        test_str = "key: !include /absolute_path.yaml"
        with self.assertRaises(OSError):
            d = flavio.io.yaml.load_include(test_str)
    def test_include_absolute_path(self):
        test_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'test', '2.yaml')
        test_str = "key: !include {}".format(test_file)
        d = flavio.io.yaml.load_include(test_str)
        self.assertDictEqual(d, {'key': {'key2': 0}})
