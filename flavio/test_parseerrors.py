import unittest
import flavio

class TestParseErrors(unittest.TestCase):
    def test_parse_errors(self):
        pul = flavio._parse_errors._pattern_upperlimit
        # just assure this matches something
        self.assertTrue(pul.match('< 3.6e-9 @90% CL'))
        self.assertTrue(pul.match(' <38 @  32.2 % C.L. '))
        self.assertTrue(pul.match('< 3.e91@ 60 % C L '))
