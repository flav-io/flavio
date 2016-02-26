import unittest
from math import radians,asin,degrees
import cmath
from flavio.physics.ckm import *
import numpy as np

# some values close to the real ones
Vus = 0.22
Vub = 3.5e-3
Vcb = 4.0e-2
gamma = radians(70.) # 70° in radians

# converting to other parametrizations
t12 = asin(Vus)
t13 = asin(Vub)
t23 = asin(Vcb)/cos(t13)
delta = gamma
laC = Vus
A = sin(t23)/laC**2
rho_minus_i_eta = sin(t13) * cmath.exp(-1j*delta) / (A*laC**3)
rho = rho_minus_i_eta.real
eta = -rho_minus_i_eta.imag
rhobar = rho*(1 - laC**2/2.)
etabar = eta*(1 - laC**2/2.)

class TestCKM(unittest.TestCase):
    v_s = ckm_standard(t12, t13, t23, delta)
    v_w = ckm_wolfenstein(laC, A, rhobar, etabar)
    v_t = ckm_tree(Vus, Vub, Vcb, gamma)
    par_s = dict(t12=t12,t13=t13,t23=t23,delta=delta)
    par_w = dict(laC=laC,A=A,rhobar=rhobar,etabar=etabar)
    par_t = dict(Vus=Vus,Vub=Vub,Vcb=Vcb,gamma=gamma)

    def test_ckm_parametrizations(self):
        np.testing.assert_almost_equal(self.v_t/self.v_s, np.ones((3,3)), decimal=5)
        np.testing.assert_almost_equal(self.v_t/self.v_w, np.ones((3,3)), decimal=5)

    def test_ckm_unitarity(self):
        np.testing.assert_almost_equal(np.dot(self.v_t,self.v_t.conj().T), np.eye(3), decimal=15)
        np.testing.assert_almost_equal(np.dot(self.v_w,self.v_w.conj().T), np.eye(3), decimal=15)
        np.testing.assert_almost_equal(np.dot(self.v_s,self.v_s.conj().T), np.eye(3), decimal=15)

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
