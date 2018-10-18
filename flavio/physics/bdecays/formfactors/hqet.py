"""Common functions for HQET form factors."""

from math import sqrt, log, pi
from functools import lru_cache
from flavio.math.functions import li2
from flavio.physics.running import running


def get_hqet_parameters(par, scale):
    p = {}
    alphas = running.get_alpha(par, scale, nf_out=5)['alpha_s']
    p['ash'] = alphas / pi
    p['mb'] = running.get_mb_pole(par)
    p['mc'] = p['mb'] - 3.4
    p['mb1S'] = running.get_mb_1S(par)
    mBbar = (par['m_B0'] + 3 * par['m_B*0']) / 4
    # eq. (25); note the comment about the renormalon cancellation thereafter
    p['Lambdabar'] = mBbar - p['mb1S'] + par['lambda_1'] / (2 * p['mb1S'])
    p['epsc'] = p['Lambdabar'] / (2 * p['mc'])
    p['epsb'] = p['Lambdabar'] / (2 * p['mb'])
    p['zc'] = p['mc'] / p['mb']
    return p

def L(par, w):
    chi2 = par['chi_2(1)'] + par['chi_2p(1)'] * (w - 1)
    chi3 = par['chi_3p(1)'] * (w - 1)
    eta = par['eta(1)'] + par['etap(1)'] * (w - 1)
    d = {}
    d[1] = -4 * (w - 1) * chi2 + 12 * chi3
    d[2] = -4 * chi3
    d[3] = 4 * chi2
    d[4] = 2 * eta - 1
    d[5] = -1
    d[6] = -2 * (1 + eta) / (w + 1)
    return d


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
