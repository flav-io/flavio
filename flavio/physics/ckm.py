"""Functions needed for the CKM matrix as well as for frequently used
combinations of CKM elements.

See the documentation of the ckmutil package for details on some of the
functions:

https://davidmstraub.github.io/ckmutil/
"""

from math import cos,sin
from cmath import exp,sqrt,phase
import numpy as np
from functools import lru_cache
from flavio.classes import AuxiliaryQuantity, Implementation
import ckmutil.ckm

# cached and/or vectorized versions of ckmutil functions
@lru_cache(maxsize=2)
def ckm_standard(t12, t13, t23, delta):
    return ckmutil.ckm.ckm_standard(t12, t13, t23, delta)

@np.vectorize
def tree_to_wolfenstein(Vus, Vub, Vcb, gamma):
    return ckmutil.ckm.tree_to_wolfenstein(Vus, Vub, Vcb, gamma)

@lru_cache(maxsize=2)
def ckm_wolfenstein(laC, A, rhobar, etabar):
    return ckmutil.ckm.ckm_wolfenstein(laC, A, rhobar, etabar)

@lru_cache(maxsize=2)
def ckm_tree(Vus, Vub, Vcb, gamma):
    return ckmutil.ckm.ckm_tree(Vus, Vub, Vcb, gamma)

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
    return AuxiliaryQuantity['CKM matrix'].prediction(par_dict=par_dict, wc_obj=None)


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
