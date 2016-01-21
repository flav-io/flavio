from math import sqrt,pi,log
import pkgutil
import numpy as np
from io import StringIO
import scipy.interpolate
from flavio.physics.running import running
from flavio.physics import ckm

"""Common functions needed for B decays."""

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
('B0','K*0'): 'bs',
('B+','K*+'): 'bs',
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
('B0','rho0'): 'B->rho',
('B+','rho+'): 'B->rho',
('B0','rho+'): 'B->rho',
('B+','rho0'): 'B->rho',
}
