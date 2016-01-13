from math import cos,sin
from cmath import exp,sqrt,phase
import numpy as np

"""Functions needed for the CKM matrix as well as for frequently used
combinations of CKM elements."""

def ckm_standard(t12, t13, t23, delta):
    r"""CKM matrix in the standard parametrization and standard phase
    convention.

    Parameters
    ----------
    t12 : float
        CKM angle $\theta_{12}$ in radians
    t13 : float
        CKM angle $\theta_{13}$ in radians
    t23 : float
        CKM angle $\theta_{23}$ in radians
    delta : float
        CKM phase $\delta=\gamma$ in radians
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
    laC : float
        Wolfenstein parameter $\lambda$ (sine of Cabibbo angle)
    A : float
        Wolfenstein parameter A
    rhobar : float
        Wolfenstein parameter $\bar\rho = \rho(1-\lambda^2/2)$
    etabar : float
        Wolfenstein parameter $\bar\eta = \eta(1-\lambda^2/2)$
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

def ckm_tree(Vus, Vub, Vcb, gamma):
    """CKM matrix in the tree parametrization and standard phase
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
    Vus : float
        CKM matrix element $V_{us}$
    Vub : float
        Absolute value of CKM matrix element $|V_{ub}|$
    Vcb : float
        CKM matrix element $V_{cb}$
    gamma : float
        CKM phase $\gamma=\delta$ in radians
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

def get_ckm(par):
    if 'laC' and 'A' and 'rhobar' and 'etabar' in par.keys():
        return ckm_wolfenstein(par['laC'], par['A'], par['rhobar'], par['etabar'])
    elif 'Vus' and 'Vub' and 'Vcb' and 'gamma' in par.keys():
        return ckm_tree(par['Vus'], par['Vub'], par['Vcb'], par['gamma'])
    elif 't12' and 't13' and 't23' and 'delta' in par.keys():
        return ckm_standard(par['t12'], par['t13'], par['t23'], par['delta'])
    else:
        raise InputError("Input parameters for CKM matrix not found.")

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

q_dict_u = {'u': 0, 'c': 1, 't': 2}
q_dict_d = {'d': 0, 's': 1, 'b': 2}

def xi(a, bc):
    r"""Returns the CKM combination $\xi_a^{bc} = V_{ab}V_{ac}^*$ if `a` is
      in `['u','c','t']` or $\xi_a^{bc} = V_{ba}V_{ca}^*$ if `a` is in
      `['d','s','b']`.

      Parameters
      ----------
      a : string
          should be either one of `['u','c','t']` or one of `['d','s','b']`
      bc : string
          should be a two-letter string with two up-type or down-type flavours,
          e.g. `'cu'`, `'bs'`. If `a` is up-type, `bc` must be down-type and vice
          versa.

      Returns
      -------
      A function that takes a parameter dictionary as input, just like `get_ckm`.
    """
    if a in q_dict_u:
        kl_num = q_dict_u[a]
        i_num = q_dict_d[bc[0]]
        j_num = q_dict_d[bc[1]]
        return lambda par: xi_kl_ij(par, kl_num, kl_num, i_num, j_num)
    else:
        ij_num = q_dict_d[a]
        k_num = q_dict_u[bc[0]]
        l_num = q_dict_u[bc[1]]
        return lambda par: xi_kl_ij(par, k_num, l_num, ij_num, ij_num)
