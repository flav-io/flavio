r"""Common functions needed for $B$ decays."""

from math import sqrt,pi,log
import pkgutil
import numpy as np
from io import StringIO
import scipy.interpolate
from flavio.physics.running import running
from flavio.physics import ckm


def lambda_K(a,b,c):
    r"""Källén function $\lambda$.

    $\lambda(a,b,c) = a^2 + b^2 + c^2 - 2 (ab + bc + ac)$
    """
    return a**2 + b**2 + c**2 - 2*(a*b + b*c + a*c)

def beta_l(ml, q2):
    if q2 == 0:
        return 0.
    return sqrt(1. - (4*ml**2)/q2)

meson_quark = {
('B+','K+'): 'bs',
('B0','K0'): 'bs',
('B0','K*0'): 'bs',
('B+','K*+'): 'bs',
('Lambdab','Lambda'): 'bs',
('B0','pi0'): 'bd',
('B+','pi+'): 'bd',
('B0','rho0'): 'bd',
('B+','rho+'): 'bd',
('B0','omega'): 'bd',
('Bs','K+'): 'bu',
('Bs','K*+'): 'bu',
('B0','pi+'): 'bu',
('B0','rho+'): 'bu',
('B0','pi+'): 'bu',
('B+','pi0'): 'bu',
('B+','rho0'): 'bu',
('B+','omega'): 'bu',
('B0','D+'): 'bc',
('B+','D0'): 'bc',
('B0','D*+'): 'bc',
('B+','D*0'): 'bc',
('Bs','phi'): 'bs',
'Bs': 'bs',
'B0': 'bs',
'B+': 'bu',
'K+': 'su',
'pi+': 'du',
}

meson_spectator = {
('B+','K+'): 'u',
('B0','K0'): 'd',
('B+','K*+'): 'u',
('B0','K*0'): 'd',
('Bs','phi'): 's',
}

quark_charge = {
'u':  2/3.,
'd': -1/3.,
's': -1/3.,
}

meson_ff = {
('B+','K+'): 'B->K',
('B0','K+'): 'B->K',
('B+','K0'): 'B->K',
('B0','K0'): 'B->K',
('B0','K*0'): 'B->K*',
('B+','K*+'): 'B->K*',
('B0','K*+'): 'B->K*',
('B+','K*0'): 'B->K*',
('Bs','K+'): 'Bs->K',
('Bs','K0'): 'Bs->K',
('Bs','K*+'): 'Bs->K*',
('Bs','K*0'): 'Bs->K*',
('Bs','phi'): 'Bs->phi',
('B0','rho0'): 'B->rho',
('B+','rho+'): 'B->rho',
('B0','rho+'): 'B->rho',
('B+','rho0'): 'B->rho',
('B+','omega'): 'B->omega',
('B0','omega'): 'B->omega',
('B0','D+'): 'B->D',
('B+','D0'): 'B->D',
('B0','D*+'): 'B->D*',
('B+','D*0'): 'B->D*',
('B0','pi+'): 'B->pi',
('B+','pi0'): 'B->pi',
('B0','pi0'): 'B->pi',
('B+','pi+'): 'B->pi',
}
