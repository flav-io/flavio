from math import sqrt
import pkgutil
import numpy as np
from io import StringIO
import scipy.interpolate

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

wcsm = {
'C7eff': -0.2909,
'C8eff': -0.1596,
'C9': 4.062,
'C10': -4.189,
}

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



def YC9(q2):
    #FIXME
    return 0.

# NNLO matrix elements of C_1 and C_2 needed for semi-leptonic B decays

_f_string = pkgutil.get_data('flavio.physics', 'data/arXiv-0810-4077v3/f_12_79.dat')
_f_array = np.loadtxt(StringIO(_f_string.decode('utf-8')))
_f_x = _f_array[::51*11,0]
_f_y = _f_array[:51*11:51,1]
_f_z = _f_array[:51,2]
_f_val_17 = _f_array[:,3].reshape(11,11,51) + 1j*_f_array[:,4].reshape(11,11,51)
_f_val_19 = _f_array[:,5].reshape(11,11,51) + 1j*_f_array[:,6].reshape(11,11,51)
_f_val_27 = _f_array[:,7].reshape(11,11,51) + 1j*_f_array[:,8].reshape(11,11,51)
_f_val_29 = _f_array[:,9].reshape(11,11,51) + 1j*_f_array[:,10].reshape(11,11,51)
_F_17 = scipy.interpolate.RegularGridInterpolator((_f_x, _f_y, _f_z), _f_val_17)
_F_19 = scipy.interpolate.RegularGridInterpolator((_f_x, _f_y, _f_z), _f_val_19)
_F_27 = scipy.interpolate.RegularGridInterpolator((_f_x, _f_y, _f_z), _f_val_27)
_F_29 = scipy.interpolate.RegularGridInterpolator((_f_x, _f_y, _f_z), _f_val_29)

def F_17(muh, z, sh):
    """Function $F_1^{(7)}$ as defined in arXiv:0810.4077.

    `muh` is $\hat\mu=\mu/m_b$,
    `z` is $z=m_c^2/m_b^2$,
    `sh` is $\hat s=q^2/m_b^2$.
    """
    return _F_17([muh, z, sh])[0]
def F_19(muh, z, sh):
    """Function $F_1^{(9)}$ as defined in arXiv:0810.4077.

    `muh` is $\hat\mu=\mu/m_b$,
    `z` is $z=m_c^2/m_b^2$,
    `sh` is $\hat s=q^2/m_b^2$.
    """
    return _F_19([muh, z, sh])[0]
def F_27(muh, z, sh):
    """Function $F_2^{(7)}$ as defined in arXiv:0810.4077.

    `muh` is $\hat\mu=\mu/m_b$,
    `z` is $z=m_c^2/m_b^2$,
    `sh` is $\hat s=q^2/m_b^2$.
    """
    return _F_27([muh, z, sh])[0]
def F_29(muh, z, sh):
    """Function $F_2^{(9)}$ as defined in arXiv:0810.4077.

    `muh` is $\hat\mu=\mu/m_b$,
    `z` is $z=m_c^2/m_b^2$,
    `sh` is $\hat s=q^2/m_b^2$.
    """
    return _F_29([muh, z, sh])[0]
