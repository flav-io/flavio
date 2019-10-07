"""Common functions for HQET form factors."""

from math import sqrt, log, pi
from functools import lru_cache
from flavio.math.functions import li2
from flavio.physics.running import running
from flavio.physics.bdecays.formfactors import common


def get_hqet_parameters(par):
    p = {}
    # The scale here is fixed to 2.7 GeV ~ sqrt(m_b^pole * m_c^pole)
    alphas = running.get_alpha(par, scale=2.7, nf_out=5)['alpha_s']
    p['ash'] = alphas / pi
    p['mb1S'] = running.get_mb_1S(par)
    p['mb'] = p['mb1S'] * (1 + 2 * alphas**2 / 9)
    p['mc'] = p['mb'] - 3.4
    mBbar = (par['m_B0'] + 3 * par['m_B*0']) / 4
    # eq. (25); note the comment about the renormalon cancellation thereafter
    p['Lambdabar'] = mBbar - p['mb'] + par['lambda_1'] / (2 * p['mb1S'])
    p['epsc'] = p['Lambdabar'] / (2 * p['mc'])
    p['epsb'] = p['Lambdabar'] / (2 * p['mb'])
    p['zc'] = p['mc'] / p['mb']
    return p

def xi(z, rho2, c, xi3, order_z):
    r"""Leading-order Isgur-Wise function:
    
    $$\xi(z)=1-\rho^2 (w-1) + c (w-1)^2 + \xi^{(3)} (w-1)^3/6
    
    where w=w(z) is expanded in $z$ up to an including terms of order
    `z**order_z`.
    """
    xi = (1
          - rho2    * common.w_minus_1_pow_n(z, n=1, order_z=order_z)
          + c       * common.w_minus_1_pow_n(z, n=2, order_z=order_z)
          + xi3 / 6 * common.w_minus_1_pow_n(z, n=3, order_z=order_z))
    return xi


def Lz(par, w, z, order_z):
    w_minus_1    = common.w_minus_1_pow_n(z, n=1, order_z=order_z)
    w_minus_1_sq = common.w_minus_1_pow_n(z, n=2, order_z=order_z)
    chi2 = par['chi_2(1)'] + par['chi_2p(1)'] * w_minus_1 + par['chi_2pp(1)'] / 2 * w_minus_1_sq
    chi3 = par['chi_3p(1)'] * w_minus_1 + par['chi_3pp(1)'] / 2 * w_minus_1_sq
    eta = par['eta(1)'] + par['etap(1)'] * w_minus_1 + par['etapp(1)'] / 2 * w_minus_1_sq
    d = {}
    # w is not expanded in the kinematical factors
    d[1] = -4 * (w - 1) * chi2 + 12 * chi3
    d[2] = -4 * chi3
    d[3] = 4 * chi2
    d[4] = 2 * eta - 1
    d[5] = -1
    d[6] = -2 * (1 + eta) / (w + 1)
    return d


def ell_i(i, par, z, order_z):
    """Sub-sub-leading power correction $\ell_i(w(z))$."""
    w_minus_1 = common.w_minus_1_pow_n(z, n=1, order_z=order_z)
    return par['CLN l_{}(1)'.format(i)] + w_minus_1 * par['CLN lp_{}(1)'.format(i)]


def ell(par, z, order_z):
    """Sub-sub-leading power correction $\ell_{i}(w(z))$ for $i=1\ldots6$ as dictionary."""
    return {i + 1: ell_i(i + 1, par, z, order_z) for i in range(6)}



def r(w):
    if w == 1:
        return 1
    return log(w + sqrt(-1 + w**2)) / sqrt(-1 + w**2)


def omega_plus(w):
    return w + sqrt(-1 + w**2)


def omega_minus(w):
    return w - sqrt(-1 + w**2)


@lru_cache(maxsize=32)
def omega(w, z):
    if w == 1:
        return -1 + (z + 1) / (z - 1) * log(z)
    return (1 + (w * (2 * li2(1 - z * omega_minus(w))
                 - li2(1 - omega_minus(w)**2) -
                 2 * li2(1 - z * omega_plus(w)) + li2(1 - omega_plus(w)**2))
                 ) / (2. * sqrt(-1 + w**2)) - w * log(z) * r(w))


def CP(w, z):
    wz = 1 / 2 * (z + 1 / z)
    return ((-2 * (-w + wz) * (-1 + z) * z *
             (-1 + z + z * (1 + z) * log(z)) *
             ((-1 + z**2) * log(z) +
              (z * (3 + z**2) +
               w *
                 (-1 + z - (3 + 2 * w) * z**2 +
                  z**3)) * r(w)) +
             4 * (w - wz)**2 * z**2 * omega(w, z)) /
            (2. * (w - wz)**2 * z**2))


def CV1(w, z):
    wz = 1 / 2 * (z + 1 / z)
    return ((12 * (-w + wz) * z -
             (-1 + z**2) * log(z) +
             2 * (1 + w) * (-1 + (-1 + 3 * w) * z - z**2) *
             r(w) + 4 * (w - wz) * z * omega(w, z)) /
            (6. * (w - wz) * z))


def CV2(w, z):
    wz = 1 / 2 * (z + 1 / z)
    return (-(z *
              (2 * (-w + wz) * (-1 + z) +
               (3 - 2 * w - (-2 + 4 * w) * z + z**2) *
                  log(z)) +
              (2 - (-1 + 5 * w + 2 * w**2) * z +
                  (2 * w + 4 * w**2) * z**2 -
                  (1 + w) * z**3) * r(w)) /
            (6. * (w - wz)**2 * z**2))


def CV3(w, z):
    wz = 1 / 2 * (z + 1 / z)
    return ((2 * (-w + wz) * (-1 + z) * z +
             (1 + (2 - 4 * w) * z + (3 - 2 * w) * z**2) *
             log(z) +
             (1 + w - (2 * w + 4 * w**2) * z +
              (-1 + 5 * w + 2 * w**2) * z**2 - 2 * z**3)
             * r(w)) / (6. * (w - wz)**2 * z))


def CA1(w, z):
    wz = 1 / 2 * (z + 1 / z)
    return ((12 * (-w + wz) * z -
             (-1 + z**2) * log(z) +
             2 * (-1 + w) * (-1 + (1 + 3 * w) * z - z**2) *
             r(w) + 4 * (w - wz) * z * omega(w, z)) /
            (6. * (w - wz) * z))


def CA2(w, z):
    wz = 1 / 2 * (z + 1 / z)
    return (-(z *
              (2 * (-w + wz) * (1 + z) +
               (3 + 2 * w - (2 + 4 * w) * z + z**2) *
                  log(z)) +
              (2 + (-1 - 5 * w + 2 * w**2) * z +
                  (-2 * w + 4 * w**2) * z**2 +
                  (1 - w) * z**3) * r(w)) /
            (6. * (w - wz)**2 * z**2))


def CA3(w, z):
    wz = 1 / 2 * (z + 1 / z)
    return ((2 * (-w + wz) * z * (1 + z) -
             (1 - (2 + 4 * w) * z + (3 + 2 * w) * z**2) *
             log(z) +
             (1 - w + (-2 * w + 4 * w**2) * z +
              (-1 - 5 * w + 2 * w**2) * z**2 + 2 * z**3)
             * r(w)) / (6. * (w - wz)**2 * z))


def CT1(w, z):
    wz = 1 / 2 * (z + 1 / z)
    return (((-1 + w) *
             (-1 + (2 + 4 * w) * z - z**2) * r(w) +
             (6 * (-w + wz) * z -
              (-1 + z**2) * log(z)) +
             2 * (w - wz) * z * omega(w, z)) / (3. * (w - wz) * z))


def CT2(w, z):
    wz = 1 / 2 * (z + 1 / z)
    return (2 * (z * log(z) + (1 - w * z) * r(w))) / (3. * (w - wz) * z)


def CT3(w, z):
    wz = 1 / 2 * (z + 1 / z)
    return (2 * (log(z) + (w - z) * r(w))) / (3. * (w - wz))
