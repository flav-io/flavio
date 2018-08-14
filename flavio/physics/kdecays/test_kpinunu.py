import unittest
import numpy as np
from .kpinunu import *
import flavio

constraints = flavio.default_parameters
wc_obj = flavio.WilsonCoefficients()
constraints_no_deltaPcu = constraints.copy()
constraints_no_deltaPcu.set_constraint('deltaPcu', 0)

class TestKpinunu(unittest.TestCase):
    # check that the known SM values are roughly reproduced
    def testkpinunu(self):
        self.assertAlmostEqual(
            flavio.sm_prediction('BR(K+->pinunu)')/1e-10,
            1, delta=0.2)
        self.assertAlmostEqual(
            flavio.sm_prediction('BR(KL->pinunu)')/3e-11,
            1, delta=0.2)
        obs = flavio.classes.Observable['BR(K+->pinunu)']
        wc_sm = flavio.WilsonCoefficients()
        # confirm that deltaPcu leads to an enhancement of the K+->pinunu BR
        # by 6% as stated in the abstract of hep-ph/0503107
        self.assertAlmostEqual(
            obs.prediction_central(constraints, wc_sm)/obs.prediction_central(constraints_no_deltaPcu, wc_sm),
            1.06,
            delta=0.01)
