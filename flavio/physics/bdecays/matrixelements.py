from math import sqrt,pi,log,atan
import pkgutil
import numpy as np
from io import StringIO
import scipy.interpolate
from flavio.physics.running import running
from flavio.physics import ckm

# functions for C9eff

def h(s, mq, mu):
  """Fermion loop function as defined e.g. in eq. (11) of hep-ph/0106067v2."""
  if mq == 0.:
      return 8/27. + (4j*pi)/9. + (8 * log(mu))/9. - (4 * log(s))/9.
  if s == 0.:
      return -4/9. * (1 + log(mq**2/mu**2))
  z = 4 * mq**2/s
  if z > 1:
      A = atan(1/sqrt(z-1))
  else:
      A = log((1+sqrt(1-z))/sqrt(z)) - 1j*pi/2.
  return (-4/9. * log(mq**2/mu**2) + 8/27. + 4/9. * z
          -4/9. * (2 + z) * sqrt(abs(z - 1)) * A)

def Y(q2, wc, par, scale):
    """Function $Y$ that contains the contributions of the matrix
    elements of four-quark operators to the effective Wilson coefficient
    $C_9^{\mathrm{eff}}=C_9 + Y(q^2)$.

    See e.g. eq. (10) of 0811.1214v5."""
    mb = running.get_mb_pole(par)
    mc = running.get_mc_pole(par)
    F_c = 4/3.*wc['C1'] +       wc['C2'] +      6*wc['C3'] +    60*wc['C5']
    F_b =    7*wc['C3'] +  4/3.*wc['C4'] +     76*wc['C5'] + 64/3.*wc['C6']
    F_u =      wc['C3'] +  4/3.*wc['C4'] +     16*wc['C5'] + 64/3.*wc['C6']
    F_4 = 4/3.*wc['C3'] + 64/9.*wc['C5'] + 64/27.*wc['C6']
    return ( h(s=q2, mq=mc, mu=scale) * F_c
    - 1/2. * h(s=q2, mq=mb, mu=scale) * F_b
    - 1/2. * h(s=q2, mq=0., mu=scale) * F_u
    + F_4 )


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

    `muh` is $\hatmu=mu/m_b$,
    `z` is $z=m_c^2/m_b^2$,
    `sh` is $\hat s=q^2/m_b^2$.
    """
    return _F_17([muh, z, sh])[0]

def F_19(muh, z, sh):
    """Function $F_1^{(9)}$ giving the contribution of $O_9$ to the matrix element
    of $O_1$, as defined in arXiv:0810.4077.

    `muh` is $\hatmu=mu/m_b$,
    `z` is $z=m_c^2/m_b^2$,
    `sh` is $\hat s=q^2/m_b^2$.
    """
    return _F_19([muh, z, sh])[0]

def F_27(muh, z, sh):
    """Function $F_2^{(7)}$ giving the contribution of $O_7$ to the matrix element
    of $O_2$, as defined in arXiv:0810.4077.

    `muh` is $\hatmu=mu/m_b$,
    `z` is $z=m_c^2/m_b^2$,
    `sh` is $\hat s=q^2/m_b^2$.
    """
    return _F_27([muh, z, sh])[0]

def F_29(muh, z, sh):
    """Function $F_2^{(9)}$ giving the contribution of $O_9$ to the matrix element
    of $O_2$, as defined in arXiv:0810.4077.

    `muh` is $\hatmu=mu/m_b$,
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
    """Function $F_8^{(7)}$ giving the contribution of $O_7$ to the matrix element
    of $O_8$, as given in eq. (40) of hep-ph/0312063.

    `sh` is $\hat s=q^2/m_b^2$,
    `Ls` is $\ln(\hat s)$.
    """
    return (-32/9. * Lmu + 8/27. * pi**2 - 44/9. - 8/9. * 1j * pi
    + (4/3. * pi**2 - 40/3.) * sh + (32/9. * pi**2 - 316/9.) * sh**2
    + (200/27. * pi**2 - 658/9.) * sh**3 - 8/9. * Ls * (sh + sh**2 + sh**3))


def delta_C7(par, wc, q2, scale, qiqj):
    alpha_s = running.get_alpha(par, scale)['alpha_s']
    mb = running.get_mb_pole(par)
    mc = running.get_mc_pole(par)
    xi_t = ckm.xi('t', qiqj)
    xi_u = ckm.xi('u', qiqj)
    muh = scale/mb
    sh = q2/mb**2
    z = mc**2/mb**2
    Lmu = log(scale/mb)
    Ls = log(sh)
    delta_t = wc['C8eff'] * F_87(Lmu, Ls, sh) + wc['C1'] * F_17(muh, z, sh) + wc['C2'] * F_27(muh, z, sh)
    return -alpha_s/(4*pi) * delta_t

def delta_C9(par, wc, q2, scale, qiqj):
    alpha_s = running.get_alpha(par, scale)['alpha_s']
    mb = running.get_mb_pole(par)
    mc = running.get_mc_pole(par)
    xi_t = ckm.xi('t', qiqj)
    xi_u = ckm.xi('u', qiqj)
    muh = scale/mb
    sh = q2/mb**2
    z = mc**2/mb**2
    Lmu = log(scale/mb)
    Ls = log(sh)
    delta_t = wc['C8eff'] * F_89(Ls, sh) + wc['C1'] * F_19(muh, z, sh) + wc['C2'] * F_29(muh, z, sh)
    return -alpha_s/(4*pi) * delta_t
