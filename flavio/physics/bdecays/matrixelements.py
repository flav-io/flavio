from math import pi
from cmath import sqrt, log, atan
import pkgutil
import numpy as np
from io import StringIO
import scipy.interpolate
import flavio
from flavio.physics.running import running
from flavio.physics import ckm
from flavio.math.functions import li2, zeta
from functools import lru_cache
from flavio.config import config

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

def Y(q2, wc, par, scale, qiqj):
    """Function $Y$ that contains the contributions of the matrix
    elements of four-quark operators to the effective Wilson coefficient
    $C_9^{\mathrm{eff}}=C_9 + Y(q^2)$.

    See e.g. eq. (10) of 0811.1214v5."""
    mb = running.get_mb_pole(par)
    mc = running.get_mc_pole(par)
    F_c = 4/3.*wc['C1_'+qiqj] +       wc['C2_'+qiqj] +      6*wc['C3_'+qiqj] +    60*wc['C5_'+qiqj]
    F_b =    7*wc['C3_'+qiqj] +  4/3.*wc['C4_'+qiqj] +     76*wc['C5_'+qiqj] + 64/3.*wc['C6_'+qiqj]
    F_u =      wc['C3_'+qiqj] +  4/3.*wc['C4_'+qiqj] +     16*wc['C5_'+qiqj] + 64/3.*wc['C6_'+qiqj]
    F_4 = 4/3.*wc['C3_'+qiqj] + 64/9.*wc['C5_'+qiqj] + 64/27.*wc['C6_'+qiqj]
    return ( h(s=q2, mq=mc, mu=scale) * F_c
    - 1/2. * h(s=q2, mq=mb, mu=scale) * F_b
    - 1/2. * h(s=q2, mq=0., mu=scale) * F_u
    + F_4 )

# eq. (43) of hep-ph/0412400v1
def Yu(q2, wc, par, scale, qiqj):
    flavio.citations.register("Beneke:2004dp")
    mc = running.get_mc_pole(par)
    return ( (4/3.*wc['C1_'+qiqj] + wc['C2_'+qiqj])
            * ( h(s=q2, mq=mc, mu=scale) - h(s=q2, mq=0, mu=scale) ))



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
_F_17 = scipy.interpolate.RegularGridInterpolator((_f_x, _f_y, _f_z), _f_val_17, bounds_error=False, fill_value=None)
_sh_F_19 = scipy.interpolate.RegularGridInterpolator((_f_x, _f_y, _f_z), _f_val_19, bounds_error=False, fill_value=None)
_F_27 = scipy.interpolate.RegularGridInterpolator((_f_x, _f_y, _f_z), _f_val_27, bounds_error=False, fill_value=None)
_sh_F_29 = scipy.interpolate.RegularGridInterpolator((_f_x, _f_y, _f_z), _f_val_29, bounds_error=False, fill_value=None)

@lru_cache(maxsize=config['settings']['cache size'])
def F_17(muh, z, sh):
    """Function $F_1^{(7)}$ giving the contribution of $O_7$ to the matrix element
    of $O_1$, as defined in arXiv:0810.4077.

    - `muh` is $\hat \mu=mu/m_b$,
    - `z` is $z=m_c^2/m_b^2$,
    - `sh` is $\hat s=q^2/m_b^2$.
    """
    flavio.citations.register("Greub:2008cy")
    return _F_17([muh, z, sh])[0]

@lru_cache(maxsize=config['settings']['cache size'])
def F_19(muh, z, sh):
    """Function $F_1^{(9)}$ giving the contribution of $O_9$ to the matrix element
    of $O_1$, as defined in arXiv:0810.4077.

    - `muh` is $\hat \mu=mu/m_b$,
    - `z` is $z=m_c^2/m_b^2$,
    - `sh` is $\hat s=q^2/m_b^2$.
    """
    flavio.citations.register("Greub:2008cy")
    if sh == 0:
        return 0
    return _sh_F_19([muh, z, sh])[0] / sh

@lru_cache(maxsize=config['settings']['cache size'])
def F_27(muh, z, sh):
    """Function $F_2^{(7)}$ giving the contribution of $O_7$ to the matrix element
    of $O_2$, as defined in arXiv:0810.4077.

    - `muh` is $\hat \mu=mu/m_b$,
    - `z` is $z=m_c^2/m_b^2$,
    - `sh` is $\hat s=q^2/m_b^2$.
    """
    flavio.citations.register("Greub:2008cy")
    return _F_27([muh, z, sh])[0]

@lru_cache(maxsize=config['settings']['cache size'])
def F_29(muh, z, sh):
    """Function $F_2^{(9)}$ giving the contribution of $O_9$ to the matrix element
    of $O_2$, as defined in arXiv:0810.4077.

    - `muh` is $\hat \mu=mu/m_b$,
    - `z` is $z=m_c^2/m_b^2$,
    - `sh` is $\hat s=q^2/m_b^2$.
    """
    flavio.citations.register("Greub:2008cy")
    if sh == 0:
        return 0
    return _sh_F_29([muh, z, sh])[0] / sh


def F_89(Ls, sh):
    """Function $F_8^{(9)}$ giving the contribution of $O_9$ to the matrix element
    of $O_8$, as given in eq. (39) of hep-ph/0312063.

    - `sh` is $\hat s=q^2/m_b^2$,
    - `Ls` is $\ln(\hat s)$.
    """
    flavio.citations.register("Asatrian:2003vq")
    return (104/9. - 32/27. * pi**2 + (1184/27. - 40/9. * pi**2) * sh
    + (14212/135. - 32/3 * pi**2) * sh**2 + (193444/945.
    - 560/27. * pi**2) * sh**3 + 16/9. * Ls * (1 + sh + sh**2 + sh**3))

def F_87(Lmu, sh):
    """Function $F_8^{(7)}$ giving the contribution of $O_7$ to the matrix element
    of $O_8$, as given in eq. (40) of hep-ph/0312063.

    - `sh` is $\hat s=q^2/m_b^2$,
    """
    flavio.citations.register("Asatrian:2003vq")
    if sh==0.:
        return (-4*(33 + 24*Lmu + 6j*pi - 2*pi**2))/27.
    return (-32/9. * Lmu + 8/27. * pi**2 - 44/9. - 8/9. * 1j * pi
    + (4/3. * pi**2 - 40/3.) * sh + (32/9. * pi**2 - 316/9.) * sh**2
    + (200/27. * pi**2 - 658/9.) * sh**3 - 8/9. * log(sh) * (sh + sh**2 + sh**3))


# Functions for the two-loop virtual corrections to the matrix elements of
# O1,2 in b->dl+l- (also needed for doubly Cabibbo-suppressed contributions
# to b>sl+l-). Taken from hep-ph/0403185v2 (Seidel)

def acot(x):
    return pi/2.-atan(x)

@lru_cache(maxsize=config['settings']['cache size'])
def SeidelA(q2, mb, mu):
    """Function $A(s\equiv q^2)$ defined in eq. (29) of hep-ph/0403185v2.
    """
    flavio.citations.register("Seidel:2004jh")
    if q2==0:
        return 1/729. * (833 + 120j*pi - 312 * log(mb**2/mu**2))
    sh = min(q2/mb**2, 0.999)
    z = 4 / sh
    return (-(104)/(243) * log((mb**2)/(mu**2)) + (4 * sh)/(27 * (1 - sh)) *
    (li2(sh) + log(sh) * log( 1 - sh)) + (1)/(729 * (1 - sh)**2) * (6 * sh *
    (29 - 47 * sh) * log(sh) + 785 - 1600 * sh + 833 * sh**2 + 6 * pi * 1j * (20 -
    49 * sh + 47 * sh**2)) - (2)/(243 * (1 - sh)**3) * (2 * sqrt( z - 1) * (-4 +
    9 * sh - 15 * sh**2 + 4 * sh**3) * acot(sqrt(z - 1)) + 9 * sh**3 *
    log(sh)**2 + 18 * pi * 1j * sh * (1 - 2 * sh) * log(sh)) + (2 * sh)/(243 *
    (1 - sh)**4) * (36 * acot( sqrt(z - 1))**2 + pi**2 * (-4 + 9 * sh - 9 *
    sh**2 + 3 * sh**3)))

@lru_cache(maxsize=config['settings']['cache size'])
def SeidelB(q2, mb, mu):
    """Function $A(s\equiv q^2)$ defined in eq. (30) of hep-ph/0403185v2.
    """
    flavio.citations.register("Seidel:2004jh")
    sh = min(q2/mb**2, 0.999)
    z = 4 / sh
    x1 = 1/2 + 1j/2 * sqrt(z - 1)
    x2 = 1/2 - 1j/2 * sqrt(z - 1)
    x3 = 1/2 + 1j/(2 * sqrt(z - 1))
    x4 = 1/2 - 1j/(2 * sqrt(z - 1))
    return ((8)/(243 * sh) * ((4 - 34 * sh - 17 * pi * 1j * sh) *
    log((mb**2)/(mu**2)) + 8 * sh * log((mb**2)/(mu**2))**2 + 17 * sh * log(sh) *
    log((mb**2)/(mu**2))) + ((2 + sh) * sqrt( z - 1))/(729 * sh) * (-48 *
    log((mb**2)/(mu**2)) * acot( sqrt(z - 1)) - 18 * pi * log(z - 1) + 3 * 1j *
    log(z - 1)**2 - 24 * 1j * li2(-x2/x1) - 5 * pi**2 * 1j + 6 * 1j * (-9 *
    log(x1)**2 + log(x2)**2 - 2 * log(x4)**2 + 6 * log(x1) * log(x2) - 4 * log(x1) *
    log(x3) + 8 * log(x1) * log(x4)) - 12 * pi * (2 * log(x1) + log(x3) + log(x4))) -
    (2)/(243 * sh * (1 - sh)) * (4 * sh * (-8 + 17 * sh) * (li2(sh) + log(sh) *
    log(1 - sh)) + 3 * (2 + sh) * (3 - sh) * log(x2/x1)**2 + 12 * pi * (-6 - sh +
    sh**2) * acot( sqrt(z - 1))) + (2)/(2187 * sh * (1 - sh)**2) * (-18 * sh * (120 -
    211 * sh + 73 * sh**2) * log(sh) - 288 - 8 * sh + 934 * sh**2 - 692 * sh**3 + 18 *
    pi * 1j * sh * (82 - 173 * sh + 73 * sh**2)) - (4)/(243 * sh * (1 - sh)**3) *
    (-2 * sqrt( z - 1) * (4 - 3 * sh - 18 * sh**2 + 16 * sh**3 - 5 * sh**4) * acot(
    sqrt(z - 1)) - 9 * sh**3 * log(sh)**2 + 2 * pi * 1j * sh * (8 - 33 * sh + 51 *
    sh**2 - 17 * sh**3) * log( sh)) + (2)/(729 * sh * (1 - sh)**4) * (72 * (3 - 8 *
    sh + 2 * sh**2) * acot( sqrt(z - 1))**2 - pi**2 * (54 - 53 * sh - 286 * sh**2 +
    612 * sh**3 - 446 * sh**4 + 113 * sh**5)) )

def SeidelC(q2, mb, mu):
    """Function $A(s\equiv q^2)$ defined in eq. (31) of hep-ph/0403185v2.
    """
    flavio.citations.register("Seidel:2004jh")
    return (-(16)/(81) * log((q2)/(mu**2)) + (428)/(243)
            - (64)/(27) * zeta(3) + (16)/(81) * pi * 1j)

def Fu_17(q2, mb, mu):
    return -SeidelA(q2, mb, mu)

def Fu_27(q2, mb, mu):
    return -(-6 * SeidelA(q2, mb, mu))

def Fu_19(q2, mb, mu):
    return -(SeidelB(q2, mb, mu) + 4 * SeidelC(q2, mb, mu))

def Fu_29(q2, mb, mu):
    return -(-6 * SeidelB(q2, mb, mu) + 3 * SeidelC(q2, mb, mu))


def delta_C7(par, wc, q2, scale, qiqj):
    alpha_s = running.get_alpha(par, scale)['alpha_s']
    mb = running.get_mb_pole(par)
    mc = par['m_c BVgamma']
    xi_t = ckm.xi('t', qiqj)(par)
    xi_u = ckm.xi('u', qiqj)(par)
    muh = scale/mb
    sh = q2/mb**2
    z = mc**2/mb**2
    Lmu = log(scale/mb)
    # computing this once to save time
    delta_tmp = wc['C1_'+qiqj] * F_17(muh, z, sh) + wc['C2_'+qiqj] * F_27(muh, z, sh)
    delta_t = wc['C8eff_'+qiqj] * F_87(Lmu, sh) + delta_tmp
    delta_u = delta_tmp + wc['C1_'+qiqj] * Fu_17(q2, mb, scale) + wc['C2_'+qiqj] * Fu_27(q2, mb, scale)
    # note the minus sign between delta_t and delta_u. This is because of a sign
    # switch in the definition of the "Fu" functions between hep-ph/0403185
    # (used here) and hep-ph/0412400, see footnote 5 of 0811.1214.
    return -alpha_s/(4*pi) * (delta_t - xi_u/xi_t * delta_u)

def delta_C9(par, wc, q2, scale, qiqj):
    alpha_s = running.get_alpha(par, scale)['alpha_s']
    mb = running.get_mb_pole(par)
    mc = running.get_mc_pole(par)
    xi_t = ckm.xi('t', qiqj)(par)
    xi_u = ckm.xi('u', qiqj)(par)
    muh = scale/mb
    sh = q2/mb**2
    z = mc**2/mb**2
    Lmu = log(scale/mb)
    Ls = log(sh)
    # computing this once to save time
    delta_tmp = wc['C1_'+qiqj] * F_19(muh, z, sh) + wc['C2_'+qiqj] * F_29(muh, z, sh)
    delta_t = wc['C8eff_'+qiqj] * F_89(Ls, sh) + delta_tmp
    delta_u = delta_tmp + wc['C1_'+qiqj] * Fu_19(q2, mb, scale) + wc['C2_'+qiqj] * Fu_29(q2, mb, scale)
    # note the minus sign between delta_t and delta_u. This is because of a sign
    # switch in the definition of the "Fu" functions between hep-ph/0403185
    # (used here) and hep-ph/0412400, see footnote 5 of 0811.1214.
    return -alpha_s/(4*pi) * (delta_t - xi_u/xi_t * delta_u)
