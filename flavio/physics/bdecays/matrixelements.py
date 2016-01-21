from math import sqrt,pi,log
import pkgutil
import numpy as np
from io import StringIO
import scipy.interpolate
from flavio.physics.running import running
from flavio.physics import ckm



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
    """Function $F_1^{(7)}$ giving the contribution of $O_7$ to the matrix element
    of $O_1$, as defined in arXiv:0810.4077.

    `muh` is $\hat\mu=\mu/m_b$,
    `z` is $z=m_c^2/m_b^2$,
    `sh` is $\hat s=q^2/m_b^2$.
    """
    return _F_17([muh, z, sh])[0]

def F_19(muh, z, sh):
    """Function $F_1^{(9)}$ giving the contribution of $O_9$ to the matrix element
    of $O_1$, as defined in arXiv:0810.4077.

    `muh` is $\hat\mu=\mu/m_b$,
    `z` is $z=m_c^2/m_b^2$,
    `sh` is $\hat s=q^2/m_b^2$.
    """
    return _F_19([muh, z, sh])[0]

def F_27(muh, z, sh):
    """Function $F_2^{(7)}$ giving the contribution of $O_7$ to the matrix element
    of $O_2$, as defined in arXiv:0810.4077.

    `muh` is $\hat\mu=\mu/m_b$,
    `z` is $z=m_c^2/m_b^2$,
    `sh` is $\hat s=q^2/m_b^2$.
    """
    return _F_27([muh, z, sh])[0]

def F_29(muh, z, sh):
    """Function $F_2^{(9)}$ giving the contribution of $O_9$ to the matrix element
    of $O_2$, as defined in arXiv:0810.4077.

    `muh` is $\hat\mu=\mu/m_b$,
    `z` is $z=m_c^2/m_b^2$,
    `sh` is $\hat s=q^2/m_b^2$.
    """
    return _F_29([muh, z, sh])[0]


def F_89(Ls, sh):
    """Function $F_8^{(9)}$ giving the contribution of $O_9$ to the matrix element
    of $O_8$, as given in eq. (39) of hep-ph/0312063.

    `sh` is $\hat s=q^2/m_b^2$,
    `Ls` is $\ln(\hat s)$.
    """
    return (104/9. - 32/27. * pi**2 + (1184/27. - 40/9. * pi**2) * sh
    + (14212/135. - 32/3 * pi**2) * sh**2 + (193444/945.
    - 560/27. * pi**2) * sh**3 + 16/9. * Ls * (1 + sh + sh**2 + sh**3))

def F_87(Lmu, Ls, sh):
    """Function $F_8^{(9)}$ giving the contribution of $O_7$ to the matrix element
    of $O_8$, as given in eq. (40) of hep-ph/0312063.

    `sh` is $\hat s=q^2/m_b^2$,
    `Ls` is $\ln(\hat s)$.
    """
    return (-32/9. * Lmu + 8/27. * pi**2 - 44/9. - 8/9. * 1j * pi
    + (4/3. * pi**2 - 40/3.) * sh + (32/9. * pi**2 - 316/9.) * sh**2
    + (200/27. * pi**2 - 658/9.) * sh**3 - 8/9. * Ls * (sh + sh**2 + sh**3))


def delta_C7(par, wc, q2, scale, qiqj):
    alpha_s = running.get_alpha(par, scale)['alpha_s']
    mb = running.get_mb(par, scale)
    mc = running.get_mc(par, scale)
    xi_t = ckm.xi('t', qiqj)
    xi_u = ckm.xi('u', qiqj)
    muh = scale/mb
    sh = q2/mb**2
    z = mc**2/mb**2
    Lmu = log(scale/mb)
    Ls = log(sh)
    Cbar2 = wc['C2'] - wc['C1']/6.
    delta_t = wc['C8eff'] * F_87(Lmu, Ls, sh) + Cbar2 * F_27(muh, z, sh)
    return -alphas/(4*pi) * delta_t
