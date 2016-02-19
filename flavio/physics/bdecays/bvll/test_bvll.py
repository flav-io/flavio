import unittest
import numpy as np
import flavio
from flavio.physics.bdecays.formfactors.b_v import bsz_parameters

class TestBVll(unittest.TestCase):
    def test_compare_to_Davids_old_code(self):
        # this actually works to 3% and better! Still, setting it to 15%
        # here to not give any error in case of future changes to the central value of the code
        delta = 0.15
        self.assertAlmostEqual(flavio.sm_prediction("AFB(B0->K*mumu)", q2=1)/-0.15349451759556973, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("FL(B0->K*mumu)", q2=1)/0.6406629242990924, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("S5(B0->K*mumu)", q2=1)/0.22071783227496475, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("P4p(B0->K*mumu)", q2=1)/0.15140284164390483, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("P5p(B0->K*mumu)", q2=1)/0.5064923738851054, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("AFB(B0->K*mumu)", q2=6)/0.17920578512694066, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("FL(B0->K*mumu)", q2=6)/0.6746725598733865, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("S4(B0->K*mumu)", q2=6)/-0.24476949751293892, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("S5(B0->K*mumu)", q2=6)/-0.3735858481345886, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("P4p(B0->K*mumu)", q2=6)/-0.529132300413707, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("P5p(B0->K*mumu)", q2=6)/-0.807602014278806, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("AFB(B0->K*mumu)", q2=15)/0.406194755197377, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("FL(B0->K*mumu)", q2=15)/0.36807161547568185, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("S3(B0->K*mumu)", q2=15)/-0.14872271241628923, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("S4(B0->K*mumu)", q2=15)/-0.2919191029733921, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("S5(B0->K*mumu)", q2=15)/-0.34293705381921147, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("P4p(B0->K*mumu)", q2=15)/-0.6065016593226283, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("P5p(B0->K*mumu)", q2=15)/-0.7124983944730857, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("dBR/dq2(B0->K*mumu)", q2=1)/5.7131974656129313e-8, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("dBR/dq2(B0->K*mumu)", q2=6)/5.524478678779278e-8, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("dBR/dq2(B0->K*mumu)", q2=15)/7.412258847550239e-8, 1, delta=delta)
