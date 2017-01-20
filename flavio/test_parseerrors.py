import unittest
import flavio

class TestParseErrors(unittest.TestCase):
    def test_parse_limits(self):
        pul = flavio._parse_errors._pattern_upperlimit
        # just assure this matches something
        self.assertTrue(pul.match('< 3.6e-9 @90% CL'))
        self.assertTrue(pul.match(' <38 @  32.2 % C.L. '))
        self.assertTrue(pul.match('< 3.e91@ 60 % C L '))
        # check that they return the right thing
        pds = flavio._parse_errors.constraints_from_string('<3.6e-9@90%CL')
        self.assertEqual(len(pds), 1)
        self.assertEqual(pds[0].limit, 3.6e-9)
        self.assertEqual(pds[0].confidence_level, 0.9)
        pds = flavio._parse_errors.constraints_from_string(' <3.6 @ 0.2 % C. L.')
        self.assertEqual(len(pds), 1)
        self.assertEqual(pds[0].limit, 3.6)
        self.assertEqual(pds[0].confidence_level, 0.002)
        # make sure using CL above 100% raises an error:
        with self.assertRaises(ValueError):
            flavio._parse_errors.constraints_from_string('< 3.6e-9 @110% CL')
        # make sure a negative limit raises an error:
        with self.assertRaises(ValueError):
            flavio._parse_errors.constraints_from_string('< -3.6e-9 @90% CL')

    def test_parse_range(self):
        pds = flavio._parse_errors.constraints_from_string('[-1,5]')
        self.assertEqual(len(pds), 1)
        self.assertEqual(pds[0].central_value, 2)
        self.assertEqual(pds[0].half_range, 3)
        pds = flavio._parse_errors.constraints_from_string('[-1 ,5 ] * 1e15')
        self.assertEqual(pds[0].central_value, 2e15)
        self.assertEqual(pds[0].half_range, 3e15)
        with self.assertRaises(ValueError):
            # max < min
            flavio._parse_errors.constraints_from_string('[1, -5]')
        pds = flavio._parse_errors.constraints_from_string('[ -1e-2,5e-2]e15')
        self.assertEqual(pds[0].central_value, 2e13)
        self.assertEqual(pds[0].half_range, 3e13)
