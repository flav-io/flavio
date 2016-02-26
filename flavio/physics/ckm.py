"""Functions needed for the CKM matrix as well as for frequently used
combinations of CKM elements."""

from math import cos,sin
from cmath import exp,sqrt,phase
import numpy as np
from functools import lru_cache
from flavio.classes import AuxiliaryQuantity, Implementation


@lru_cache(maxsize=2)
def ckm_standard(t12, t13, t23, delta):
    r"""CKM matrix in the standard parametrization and standard phase
    convention.

    Parameters
    ----------

    - `t12`: CKM angle $\theta_{12}$ in radians
    - `t13`: CKM angle $\theta_{13}$ in radians
    - `t23`: CKM angle $\theta_{23}$ in radians
    - `delta`: CKM phase $\delta=\gamma$ in radians
    """
    c12 = cos(t12)
    c13 = cos(t13)
    c23 = cos(t23)
    s12 = sin(t12)
    s13 = sin(t13)
    s23 = sin(t23)
    return np.array([[c12*c13,
        c13*s12,
        s13/exp(1j*delta)],
        [-(c23*s12) - c12*exp(1j*delta)*s13*s23,
        c12*c23 - exp(1j*delta)*s12*s13*s23,
        c13*s23],
        [-(c12*c23*exp(1j*delta)*s13) + s12*s23,
        -(c23*exp(1j*delta)*s12*s13) - c12*s23,
        c13*c23]])

@np.vectorize
def tree_to_wolfenstein(Vus, Vub, Vcb, gamma):
    laC = Vus/sqrt(1-Vub**2)
    A = Vcb/sqrt(1-Vub**2)/laC**2
    rho = Vub*cos(gamma)/A/laC**3
    eta = Vub*sin(gamma)/A/laC**3
    rhobar = rho*(1 - laC**2/2.)
    etabar = eta*(1 - laC**2/2.)
    return laC, A, rhobar, etabar

@lru_cache(maxsize=2)
def ckm_wolfenstein(laC, A, rhobar, etabar):
    r"""CKM matrix in the Wolfenstein parametrization and standard phase
    convention.

    This function does not rely on an expansion in the Cabibbo angle but
    defines, to all orders in $\lambda$,

    - $\lambda = \sin\theta_{12}$
    - $A\lambda^2 = \sin\theta_{23}$
    - $A\lambda^3(\rho-i \eta) = \sin\theta_{13}e^{-i\delta}$

    where $\rho = \bar\rho/(1-\lambda^2/2)$ and
    $\eta = \bar\eta/(1-\lambda^2/2)$.

    Parameters
    ----------

    - `laC`: Wolfenstein parameter $\lambda$ (sine of Cabibbo angle)
    - `A`: Wolfenstein parameter $A$
    - `rhobar`: Wolfenstein parameter $\bar\rho = \rho(1-\lambda^2/2)$
    - `etabar`: Wolfenstein parameter $\bar\eta = \eta(1-\lambda^2/2)$
    """
    rho = rhobar/(1 - laC**2/2.)
    eta = etabar/(1 - laC**2/2.)
    return np.array([[sqrt(1 - laC**2)*sqrt(1 - A**2*laC**6*((-1j)*eta + rho)*((1j)*eta + rho)),
        laC*sqrt(1 - A**2*laC**6*((-1j)*eta + rho)*((1j)*eta + rho)),
        A*laC**3*((-1j)*eta + rho)],
        [-(laC*sqrt(1 - A**2*laC**4)) - A**2*laC**5*sqrt(1 - laC**2)*((1j)*eta + rho),
        sqrt(1 - laC**2)*sqrt(1 -  A**2*laC**4) - A**2*laC**6*((1j)*eta + rho),
        A*laC**2*sqrt(1 - A**2*laC**6*((-1j)*eta + rho)*((1j)*eta + rho))],
        [A*laC**3 - A*laC**3*sqrt(1 - laC**2)*sqrt(1 - A**2*laC**4)*((1j)*eta + rho),
        -(A*laC**2*sqrt(1 - laC**2)) - A*laC**4*sqrt(1 - A**2*laC**4)*((1j)*eta + rho),
        sqrt(1 - A**2*laC**4)*sqrt(1 - A**2*laC**6*((-1j)*eta + rho)*((1j)*eta + rho))]])

@lru_cache(maxsize=2)
def ckm_tree(Vus, Vub, Vcb, gamma):
    r"""CKM matrix in the tree parametrization and standard phase
    convention.

    In this parametrization, the parameters are directly measured from
    tree-level $B$ decays. It is thus particularly suited for new physics
    analyses because the tree-level decays should be dominated by the Standard
    Model. This function involves no analytical approximations.

    Relation to the standard parametrization:

    - $V_{us} = \cos \theta_{13} \sin \theta_{12}$
    - $|V_{ub}| = |\sin \theta_{13}|$
    - $V_{cb} = \cos \theta_{13} \sin \theta_{23}$
    - $\gamma=\delta$

    Parameters
    ----------

    - `Vus`: CKM matrix element $V_{us}$
    - `Vub`: Absolute value of CKM matrix element $|V_{ub}|$
    - `Vcb`: CKM matrix element $V_{cb}$
    - `gamma`: CKM phase $\gamma=\delta$ in radians
    """
    return np.array([[sqrt(1 - Vub**2)*sqrt(1 - Vus**2/(1 - Vub**2)),
        Vus,
        Vub/exp(1j*gamma)],
        [-((sqrt(1 - Vcb**2/(1 - Vub**2))*Vus)/sqrt(1 - Vub**2)) - (Vub*exp(1j*gamma)*Vcb*sqrt(1 - Vus**2/(1 - Vub**2)))/sqrt(1 - Vub**2),
        -((Vub*exp(1j*gamma)*Vcb*Vus)/(1 - Vub**2)) + sqrt(1 - Vcb**2/(1 - Vub**2))*sqrt(1 - Vus**2/(1 - Vub**2)),
        Vcb],
        [(Vcb*Vus)/(1 - Vub**2) - Vub*exp(1j*gamma)*sqrt(1 - Vcb**2/(1 - Vub**2))*sqrt(1 - Vus**2/(1 - Vub**2)),
        -((Vub*exp(1j*gamma)*sqrt(1 - Vcb**2/(1 - Vub**2))*Vus)/sqrt(1 - Vub**2)) - (Vcb*sqrt(1 - Vus**2/(1 - Vub**2)))/sqrt(1 - Vub**2),
        sqrt(1 - Vub**2)*sqrt(1 - Vcb**2/(1 - Vub**2))]])

# Auxiliary Quantity instance
a = AuxiliaryQuantity(name='CKM matrix')
a.set_description('Cabibbo-Kobayashi-Maskawa matrix in the standard phase convention')

# Implementation instances

def _func_standard(wc_obj, par):
    return ckm_standard(par['t12'], par['t13'], par['t23'], par['delta'])
def _func_tree(wc_obj, par):
    return ckm_tree(par['Vus'], par['Vub'], par['Vcb'], par['gamma'])
def _func_wolfenstein(wc_obj, par):
    return ckm_wolfenstein(par['laC'], par['A'], par['rhobar'], par['etabar'])

i = Implementation(name='Standard', quantity='CKM matrix', function=_func_standard)
i = Implementation(name='Tree', quantity='CKM matrix', function=_func_tree)
i = Implementation(name='Wolfenstein', quantity='CKM matrix', function=_func_wolfenstein)

def get_ckm(par_dict):
    return AuxiliaryQuantity.get_instance('CKM matrix').prediction(par_dict=par_dict, wc_obj=None)


def get_ckmangle_beta(par):
    r"""Returns the CKM angle $\beta$."""
    V = get_ckm(par)
    # see eq. (12.16) of http://pdg.lbl.gov/2015/reviews/rpp2014-rev-ckm-matrix.pdf
    return phase(-V[1,0]*V[1,2].conj()/V[2,0]/V[2,2].conj())

def get_ckmangle_alpha(par):
    r"""Returns the CKM angle $\alpha$."""
    V = get_ckm(par)
    # see eq. (12.16) of http://pdg.lbl.gov/2015/reviews/rpp2014-rev-ckm-matrix.pdf
    return phase(-V[2,0]*V[2,2].conj()/V[0,0]/V[0,2].conj())

def get_ckmangle_gamma(par):
    r"""Returns the CKM angle $\gamma$."""
    V = get_ckm(par)
    # see eq. (12.16) of http://pdg.lbl.gov/2015/reviews/rpp2014-rev-ckm-matrix.pdf
    return phase(-V[0,0]*V[0,2].conj()/V[1,0]/V[1,2].conj())


# Some useful shorthands for CKM combinations appearing in FCNC amplitudes
def xi_kl_ij(par, k, l, i, j):
    V = get_ckm(par)
    return V[k,i] * V[l,j].conj()

_q_dict_u = {'u': 0, 'c': 1, 't': 2}
_q_dict_d = {'d': 0, 's': 1, 'b': 2}

def xi(a, bc):
    r"""Returns the CKM combination $\xi_a^{bc} = V_{ab}V_{ac}^\ast$ if `a` is
      in `['u','c','t']` or $\xi_a^{bc} = V_{ba}V_{ca}^\ast$ if `a` is in
      `['d','s','b']`.

      Parameters
      ----------
      - `a`: should be either one of `['u','c','t']` or one of `['d','s','b']`
      - `bc`: should be a two-letter string with two up-type or down-type flavours,
          e.g. `'cu'`, `'bs'`. If `a` is up-type, `bc` must be down-type and vice
          versa.

      Returns
      -------
      A function that takes a parameter dictionary as input, just like `get_ckm`.
    """
    if a in _q_dict_u:
        kl_num = _q_dict_u[a]
        i_num = _q_dict_d[bc[0]]
        j_num = _q_dict_d[bc[1]]
        return lambda par: xi_kl_ij(par, kl_num, kl_num, i_num, j_num)
    else:
        ij_num = _q_dict_d[a]
        k_num = _q_dict_u[bc[0]]
        l_num = _q_dict_u[bc[1]]
        return lambda par: xi_kl_ij(par, k_num, l_num, ij_num, ij_num)
