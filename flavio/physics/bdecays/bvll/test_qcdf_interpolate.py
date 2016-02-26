import unittest
import numpy as np
import flavio

par = flavio.default_parameters.get_central_all()
wc_obj = flavio.WilsonCoefficients()
wc = flavio.physics.bdecays.wilsoncoefficients.wctot_dict(wc_obj, 'bsmumu', flavio.config['renormalization scale']['bvll'], par)

class TestQCDFInterpolate(unittest.TestCase):
    def test_qcdf_interpolate(self):
        q2 = 3.33333
        B = 'B0'
        V = 'K*0'
        lep = 'mu'
        amps_in = flavio.physics.bdecays.bvll.qcdf_interpolate.helicity_amps_qcdf(q2, par, B, V, lep)
        amps_ex = flavio.physics.bdecays.bvll.qcdf.helicity_amps_qcdf(q2, wc, par, B, V, lep)
        for i in amps_in.keys():
            if not amps_ex[i] == 0:
                self.assertAlmostEqual(amps_in[i]/(amps_ex[i]), 1, places=2)
            else:
                self.assertEqual(amps_in[i], 0)

        # and the same game again for different q2 and process
        q2 = 0.15
        B = 'B+'
        V = 'K*+'
        lep = 'e'
        amps_in = flavio.physics.bdecays.bvll.qcdf_interpolate.helicity_amps_qcdf(q2, par, B, V, lep)
        amps_ex = flavio.physics.bdecays.bvll.qcdf.helicity_amps_qcdf(q2, wc, par, B, V, lep)
        for i in amps_in.keys():
            if not amps_ex[i] == 0:
                self.assertAlmostEqual(amps_in[i]/(amps_ex[i]), 1, places=0)
            else:
                self.assertEqual(amps_in[i], 0)

        # and the same game again for different q2 and process
        q2 = 5.935
        B = 'Bs'
        V = 'phi'
        lep = 'mu'
        amps_in = flavio.physics.bdecays.bvll.qcdf_interpolate.helicity_amps_qcdf(q2, par, B, V, lep)
        amps_ex = flavio.physics.bdecays.bvll.qcdf.helicity_amps_qcdf(q2, wc, par, B, V, lep)
        for i in amps_in.keys():
            if not amps_ex[i] == 0:
                self.assertAlmostEqual(amps_in[i]/(amps_ex[i]), 1, places=0)
            else:
                self.assertEqual(amps_in[i], 0)
