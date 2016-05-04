import unittest
import numpy as np
import flavio

constraints = flavio.default_parameters
wc_obj = flavio.WilsonCoefficients()
par = constraints.get_central_all()

class TestAngular(unittest.TestCase):
    def test_timedependent(self):
        # self.assertAlmostEqual(
        #     flavio.sm_prediction('BR(B0->K*nunu)')/9.48e-6,
        #     1, delta=0.2)
        q2 = 3.
        B = 'B0'
        V = 'K*0'
        lep = 'mu'
        mB = par['m_'+B]
        mV = par['m_'+V]
        ml = par['m_'+lep]
        scale = 4.8
        mb = flavio.physics.running.running.get_mb(par, scale)
        H = flavio.physics.bdecays.bvll.amplitudes.helicity_amps_ff(q2, wc_obj, par, B, V, lep, cp_conjugate=False)
        # these are the angular coefficients in the "G" basis
        G = flavio.physics.bdecays.angular.angularcoeffs_general_Gbasis_v(H, q2, mB, mV, mb, 0, ml, ml)
        # these are the h coefficients of the time dependent angular distribution
        h = flavio.physics.bdecays.angular.angularcoeffs_h_Gbasis_v(0, H, H, q2, mB, mV, mb, 0, ml, ml)
        # for q/p=1 (phi=0) and Htilde = H (this is unphysical ...) the two must coincide:
        for k in G.keys():
            self.assertAlmostEqual(h[k]/G[k]/2., 1., delta=1e-10)
