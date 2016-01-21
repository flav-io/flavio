from math import sqrt
import pkgutil
import numpy as np
from io import StringIO
import scipy.interpolate
from flavio.physics.running import running
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

# SM Wilson coefficients at 120 GeV in the basis
# [ C_1, C_2, C_3, C_4, C_5, C_6,
# C_7^eff, C_8^eff,
# C_9, C_10,
# C_3^Q, C_4^Q, C_5^Q, C_6^Q,
# Cb ]
# where all operators are defined as in hep-ph/0512066 *except*
# C_9,10, which are defined with an additional alpha/4pi prefactor.
_wcsm_120 = np.array([  1.99030910e-01,   1.00285703e+00,  -4.17672471e-04,
         2.00964137e-03,   5.20961618e-05,   9.65703651e-05,
        -1.98510105e-01,  -1.09453204e-01,   1.52918563e+00,
        -4.06926405e+00,   6.15944332e-03,   0.00000000e+00,
        -1.12876870e-03,   0.00000000e+00,  -3.24099235e-03])

def wctot_dict(wc_obj, sector, scale, par):
    r"""Get a dictionary with the total (SM + new physics) values  of the
    $\Delta F=1$ Wilson coefficients at a fiven scale, given a
    WilsonCoefficients instance."""
    wc_np = wc_obj.get_wc(sector, scale, par)
    wc_sm = running.get_wilson(par, _wcsm_120, wc_obj.adm[sector], 120., scale)
    wc_labels = wc_obj.coefficients[sector]
    wc_dict =  dict(zip(wc_labels, wc_np + wc_sm))
    return wc_dict

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
