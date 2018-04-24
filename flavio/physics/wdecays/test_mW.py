import unittest
import flavio
from . import mw
from flavio.physics.zdecays.test_smeftew import ZeroDict
import wilson


par = flavio.default_parameters.get_central_all()


class TestMW(unittest.TestCase):
    def test_mW_SM(self):
        self.assertAlmostEqual(mw.mW_SM(par), 80.3779, delta=0.02)

    def test_shifts_sm(self):
        C = ZeroDict({})
        self.assertEqual(mw.dmW_SMEFT(par, C), 0)

    def test_obs(self):
        w = wilson.Wilson({}, scale=91.1876, eft='SMEFT', basis='Warsaw')
        self.assertAlmostEqual(flavio.sm_prediction('m_W'),
                               par['m_W'], delta=0.03)
        self.assertEqual(flavio.sm_prediction('m_W'),
                         flavio.np_prediction('m_W', w))
