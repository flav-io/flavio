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
