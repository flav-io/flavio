import unittest
from math import radians,asin,degrees
import cmath
from flavio.physics.ckm import *
import numpy as np
from ckmutil.ckm import gamma_to_delta

# some values close to the real ones
Vus = 0.22
Vub = 3.5e-3
Vcb = 4.0e-2
gamma = radians(70.) # 70° in radians

# converting to other parametrizations
s13 = Vub
c13 = np.sqrt(1-s13**2)
s12 = Vus/c13
c12 = np.sqrt(1-s12**2)
s23 = Vcb/c13
c23 = np.sqrt(1-s23**2)
t12 = asin(s12)
t13 = asin(s13)
t23 = asin(s23)
delta = gamma_to_delta(t12, t13, t23, gamma)
laC = Vus
A = s23/laC**2
rho_minus_i_eta = s13 * cmath.exp(-1j*delta) / (A*laC**3)
rho = rho_minus_i_eta.real
eta = -rho_minus_i_eta.imag
rhobar_plus_i_etabar = ( # e.q. Eq. (92) in arXiv:2206.07501
    sqrt(1-laC**2)*(rho + 1j*eta)/
    (sqrt(1-A**2*laC**4)+sqrt(1-laC**2)*A**2*laC**4*(rho + 1j*eta))
)
rhobar = rhobar_plus_i_etabar.real
etabar = rhobar_plus_i_etabar.imag
Vcd_complex = - s12*c23 - c12*s23*s13 * np.exp(1j*delta)
Vtd_complex = s12*s23 - c12*c23*s13 * np.exp(1j*delta)
beta = np.angle(-Vcd_complex/Vtd_complex)

class TestCKM(unittest.TestCase):
    v_s = ckm_standard(t12, t13, t23, delta)
    v_w = ckm_wolfenstein(laC, A, rhobar, etabar)
    v_t = ckm_tree(Vus, Vub, Vcb, gamma)
    v_b = ckm_beta_gamma(Vus, Vcb, beta, gamma)
    par_s = dict(t12=t12,t13=t13,t23=t23,delta=delta)
    par_w = dict(laC=laC,A=A,rhobar=rhobar,etabar=etabar)
    par_t = dict(Vus=Vus,Vub=Vub,Vcb=Vcb,gamma=gamma)

    def test_ckm_parametrizations(self):
        np.testing.assert_almost_equal(self.v_t/self.v_s, np.ones((3,3)), decimal=5)
        np.testing.assert_almost_equal(self.v_t/self.v_w, np.ones((3,3)), decimal=5)
        np.testing.assert_almost_equal(self.v_t/self.v_b, np.ones((3,3)), decimal=5)

    def test_ckm_unitarity(self):
        np.testing.assert_almost_equal(np.dot(self.v_t,self.v_t.conj().T), np.eye(3), decimal=15)
        np.testing.assert_almost_equal(np.dot(self.v_w,self.v_w.conj().T), np.eye(3), decimal=15)
        np.testing.assert_almost_equal(np.dot(self.v_s,self.v_s.conj().T), np.eye(3), decimal=15)
        np.testing.assert_almost_equal(np.dot(self.v_b,self.v_b.conj().T), np.eye(3), decimal=15)

    def test_get_ckm(self):
        # np.testing.assert_array_equal(get_ckm(self.par_s), self.v_s)
    #     np.testing.assert_array_equal(get_ckm(self.par_w), self.v_w)
        np.testing.assert_array_equal(get_ckm(self.par_t), self.v_t)

    def test_ckm_xi(self):
        # check if xi functions are properly defined
        self.assertEqual(xi_kl_ij(self.par_t, 0, 1, 2, 1),
                         np.dot(self.v_t[0,2],self.v_t[1,1].conj()))
        self.assertEqual(xi('t','bs')(self.par_t),
                         np.dot(self.v_t[2,2],self.v_t[2,1].conj()))
        self.assertEqual(xi('b','cu')(self.par_t),
                         np.dot(self.v_t[1,2],self.v_t[0,2].conj()))
        # make sure illegal flavours raise an error
        with self.assertRaises(KeyError):
            xi('b','sd')
        with self.assertRaises(KeyError):
            xi('t','cu')
        with self.assertRaises(KeyError):
          xi('x','bs')

    def test_ckm_angles(self):
        c_gamma = get_ckmangle_gamma(self.par_t)
        # angle gamma should be equal to input
        self.assertAlmostEqual(c_gamma/gamma, 1., places=3)
        c_beta = get_ckmangle_beta(self.par_t)
        c_alpha = get_ckmangle_alpha(self.par_t)
        # some of angles should be 180°
        self.assertEqual(
            degrees(c_alpha) + degrees(c_beta) + degrees(c_gamma), 180.)
