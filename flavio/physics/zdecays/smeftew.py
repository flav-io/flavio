r"""Shift of electroweak parameters and $Z$ couplings due to dimension-6
operators in SMEFT.

Taken from Appendix A.1 in arXiv:1706.08945."""

from math import sqrt, pi


# fermion quantum numbers
_QN = {}
_QN['e'] = _QN['mu'] = _QN['tau'] = {'T3L': -1 / 2, 'Q': -1, 'Nc': 1}
_QN['nue'] = _QN['numu'] = _QN['nutau'] = {'T3L': 1 / 2, 'Q': 0, 'Nc': 1}
_QN['u'] = _QN['c'] = {'T3L': 1 / 2, 'Q': 2 / 3, 'Nc': 3}
_QN['d'] = _QN['s'] = _QN['b'] = {'T3L': -1 / 2, 'Q': -1 / 3, 'Nc': 3}


_sectors = {'l': {'e': 1, 'mu': 2, 'tau': 3},
            'nu': {'nue': 1, 'numu': 2, 'nutau': 3},
            'u': {'u': 1, 'c': 2,},
            'd': {'d': 1, 's': 2, 'b': 3}}


def d_GF(par, C):
    return 1 / (sqrt(2) * par['GF']) * (sqrt(2) * (C['phil3_22'] + C['phil3_11']) / 2 - C['ll_1221'] / sqrt(2))


def _sinthetahat2(par):
    r"""$sin^2(\hat\theta)$"""
    return 1 / 2 * (1 - sqrt(1 - (4 * pi * par['alpha_e']) / (sqrt(2) * par['GF'] * par['m_Z']**2)))

def d_sh2(par, C):
    sh2 = _sinthetahat2(par)
    sh = sqrt(sh2)
    ch2 = 1 - sh2
    ch = sqrt(ch2)
    s2h = 2 * ch * sh # sin(2 * thetahat)
    c2h = ch**2 - sh**2 # cos(2 * thetahat)
    dC = s2h * (C['phiD'] + 2 * (C['phil3_22'] + C['phil3_11']) - 2 * C['ll_1221']) + 4 * C['phiWB']
    return s2h / (8 * c2h * sqrt(2) * par['GF']) * dC


def d_mZ2(par, C):
    return (1 / (2 * sqrt(2)) * par['m_Z']**2 / par['GF'] * C['phiD']
            + (2**(1 / 4) * sqrt(pi * par['alpha_e']) * par['m_Z'])
            / par['GF']**(3 / 2) * C['phiWB'])


def d_gZb(par, C):
    sh2 = _sinthetahat2(par)
    sh = sqrt(sh2)
    ch = sqrt(1 - sh2)
    return (-d_GF(par, C) / sqrt(2)
            - d_mZ2(par, C) / (2 * par['m_Z']**2)
            + sh * ch / (sqrt(2) * par['GF']) * C['phiWB'])


def gV_SM(f, par):
    T3L = _QN[f]['T3L']
    Q = _QN[f]['Q']
    if f == 'b':
        s2w_eff = par['s2w'] * 1.0065
    else:
        s2w_eff = par['s2w'] * 1.0010
    return T3L / 2 - Q * s2w_eff


def gA_SM(f, par):
    T3L = _QN[f]['T3L']
    return T3L / 2


def d_gVl(f1, f2, par, C):
    i, j = sorted((_sectors['l'][f1], _sectors['l'][f2]))
    d_g = -(C['phie_{}{}'.format(i, j)] + C['phil1_{}{}'.format(i, j)] + C['phil3_{}{}'.format(i, j)]) / (4 * sqrt(2) * par['GF'])
    if i > j:
        d_g = d_g.conjugate()
    if f1 == f2:
        d_g += d_gZb(par, C) * gV_SM(f1, par) + d_sh2(par, C)
    return d_g


def d_gVnu(f1, f2, par, C):
    i, j = sorted((_sectors['nu'][f1], _sectors['nu'][f2]))
    d_g = -(C['phil1_{}{}'.format(i, j)] - C['phil3_{}{}'.format(i, j)]) / (4 * sqrt(2) * par['GF'])
    if i > j:
        d_g = d_g.conjugate()
    if f1 == f2:
        d_g += d_gZb(par, C) * gV_SM(f1, par)
    return d_g


def d_gVu(f1, f2, par, C):
    i, j = sorted((_sectors['u'][f1], _sectors['u'][f2]))
    d_g = (-C['phiu_{}{}'.format(i, j)] - C['phiq1_{}{}'.format(i, j)] + C['phiq3_{}{}'.format(i, j)]) / (4 * sqrt(2) * par['GF'])
    if i > j:
        d_g = d_g.conjugate()
    if f1 == f2:
        d_g += d_gZb(par, C) * gV_SM(f1, par) - 2 / 3 * d_sh2(par, C)
    return d_g


def d_gVd(f1, f2, par, C):
    i, j = sorted((_sectors['d'][f1], _sectors['d'][f2]))
    d_g = -(C['phid_{}{}'.format(i, j)] + C['phiq1_{}{}'.format(i, j)] + C['phiq3_{}{}'.format(i, j)]) / (4 * sqrt(2) * par['GF'])
    if i > j:
        d_g = d_g.conjugate()
    if f1 == f2:
        d_g += d_gZb(par, C) * gV_SM(f1, par) + 1 / 3 * d_sh2(par, C)
    return d_g


def d_gAl(f1, f2, par, C):
    i, j = sorted((_sectors['l'][f1], _sectors['l'][f2]))
    d_g = (C['phie_{}{}'.format(i, j)] - C['phil1_{}{}'.format(i, j)] - C['phil3_{}{}'.format(i, j)]) / (4 * sqrt(2) * par['GF'])
    if i > j:
        d_g = d_g.conjugate()
    if f1 == f2:
        d_g += d_gZb(par, C) * gA_SM(f1, par)
    return d_g


def d_gAnu(f1, f2, par, C):
    return d_gVnu(f1, f2, par, C)  # V == A for neutrinos


def d_gAu(f1, f2, par, C):
    i, j = sorted((_sectors['u'][f1], _sectors['u'][f2]))
    d_g = -(-C['phiu_{}{}'.format(i, j)] + C['phiq1_{}{}'.format(i, j)] - C['phiq3_{}{}'.format(i, j)]) / (4 * sqrt(2) * par['GF'])
    if i > j:
        d_g = d_g.conjugate()
    if f1 == f2:
        d_g += d_gZb(par, C) * gA_SM(f1, par)
    return d_g


def d_gAd(f1, f2, par, C):
    i, j = sorted((_sectors['d'][f1], _sectors['d'][f2]))
    d_g = (C['phid_{}{}'.format(i, j)] - C['phiq1_{}{}'.format(i, j)] - C['phiq3_{}{}'.format(i, j)]) / (4 * sqrt(2) * par['GF'])
    if i > j:
        d_g = d_g.conjugate()
    if f1 == f2:
        d_g += d_gZb(par, C) * gA_SM(f1, par)
    return d_g


def d_gV(f1, f2, par, C):
    if f1 in _sectors['l']:
        return d_gVl(f1, f2, par, C)
    if f1 in _sectors['nu']:
        return d_gVnu(f1, f2, par, C)
    if f1 in _sectors['u']:
        return d_gVu(f1, f2, par, C)
    if f1 in _sectors['d']:
        return d_gVd(f1, f2, par, C)


def d_gA(f1, f2, par, C):
    if f1 in _sectors['l']:
        return d_gAl(f1, f2, par, C)
    if f1 in _sectors['nu']:
        return d_gAnu(f1, f2, par, C)
    if f1 in _sectors['u']:
        return d_gAu(f1, f2, par, C)
    if f1 in _sectors['d']:
        return d_gAd(f1, f2, par, C)


def d_gWl(f1, f2, par, C):
    i, j = sorted((_sectors['l'][f1], _sectors['l'][f2]))
    sh2 = _sinthetahat2(par)
    sh = sqrt(sh2)
    ch = sqrt(1 - sh2)
    d_g = (C['phil3_{}{}'.format(i, j)] + 1 / 2 * ch / sh * C['phiWB']) / (2 * sqrt(2) * par['GF']) - 1 / 4 * d_sh2(par, C) / sh**2
    return d_g


def d_gWq(f1, f2, par, C):
    i, j = sorted((_sectors['u'][f1], _sectors['d'][f2]))
    sh2 = _sinthetahat2(par)
    sh = sqrt(sh2)
    ch = sqrt(1 - sh2)
    d_g = (C['phiq3_{}{}'.format(i, j)] + 1 / 2 * ch / sh * C['phiWB']) / (2 * sqrt(2) * par['GF']) - 1 / 4 * d_sh2(par, C) / sh**2
    return d_g


def d_gW(f1, f2, par, C):
    if f1 in _sectors['l']:
        return d_gWl(f1, f2, par, C)
    if f1 in _sectors['u']:
        return d_gWq(f1, f2, par, C)
