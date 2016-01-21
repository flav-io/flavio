from math import sqrt,pi
import numpy as np
from flavio.physics.bdecays.common import lambda_K, beta_l, meson_quark, meson_ff, YC9, wctot_dict
from flavio.physics import ckm
from flavio.physics.bdecays.formfactors import FormFactorParametrization as FF
from flavio.config import config
from flavio.physics.running import running

"""Functions for exclusive $B\to V\ell^+\ell^-$ decays."""


def prefactor(q2, par, B, V, lep):
    GF = par['Gmu']
    scale = config['bdecays']['scale_bvll']
    alphaem = running.get_alpha(par, scale)['alpha_e']
    ml = par[('mass',lep)]
    mB = par[('mass',B)]
    mV = par[('mass',V)]
    tauB = par[('lifetime',B)]
    X = sqrt(lambda_K(mB**2,q2,mV**2))/2.
    di_dj = meson_quark[(B,V)]
    xi_t = ckm.xi('t',di_dj)(par)
    if q2 <= 4*ml**2:
        return 0
    return ( sqrt((GF**2 * alphaem**2)/(3 * 2**10 * pi**5 * mB**3)
            * q2 * 2 * X *beta_l(ml, q2)) * xi_t )

def transversity_amps(q2, wc, par, B, V, lep):
    ml = par[('mass',lep)]
    mB = par[('mass',B)]
    scale = config['bdecays']['scale_bvll']
    mb = running.get_mb(par, scale)
    mV = par[('mass',V)]
    X = sqrt(lambda_K(mB**2,q2,mV**2))/2.
    N = prefactor(q2, par, B, V, lep)
    ta = {}
    c7pl = wc['C7eff'] + wc['C7effp']
    c7mi = wc['C7eff'] - wc['C7effp']
    c9pl = YC9(q2) + wc['C9'] + wc['C9p']
    c9mi = YC9(q2) + wc['C9'] - wc['C9p']
    c10pl = wc['C10'] + wc['C10p']
    c10mi = wc['C10'] - wc['C10p']
    csmi = wc['CS'] - wc['CSp']
    cpmi = wc['CP'] - wc['CPp']
    ff = FF.parametrizations['bsz3'].get_ff(meson_ff[(B,V)], q2, par)
    ta['perp_L'] = N * sqrt(2)*2*X * ((c9pl - c10pl) * (ff['V']/(mB + mV)) + 2*mb/q2 * c7pl * ff['T1'])
    ta['perp_R'] = N * sqrt(2)*2*X * ((c9pl + c10pl) * (ff['V']/(mB + mV)) + 2*mb/q2 * c7pl * ff['T1'])
    ta['para_L'] = -N * sqrt(2) * (mB**2 - mV**2) * ((c9mi - c10mi) * (ff['A1']/(mB - mV)) + 2*mb/q2 * c7mi * ff['T2'])
    ta['para_R'] = -N * sqrt(2) * (mB**2 - mV**2) * ((c9mi + c10mi) * (ff['A1']/(mB - mV)) + 2*mb/q2 * c7mi * ff['T2'])
    ta['0_L'] = -N * (1/(2 * mV * sqrt(q2))) * ((c9mi - c10mi) * 16*mB*mV**2 * ff['A12']
                + 2 * mb * c7mi * 8*mB*mV**2/(mB+mV) * ff['T23'])
    ta['0_R'] = -N * (1/(2 * mV * sqrt(q2))) * ((c9mi + c10mi) * 16*mB*mV**2 * ff['A12']
                + 2 * mb * c7mi * 8*mB*mV**2/(mB+mV) * ff['T23'])
    ta['t'] = N * 1/sqrt(q2) * 2 * X * (cpmi) * ff['A0']
    ta['S'] = -N * 2*X * csmi * ff['A0']
    return ta


def transversity_amps_bar(q2, wc, par, B, V, lep):
    # FIXME need to implement CP conjugation
    return transversity_amps(q2, wc, par, B, V, lep)

def angulardist(transversity_amps, q2, par, lep):
    r"""Returns the angular coefficients of the 4-fold differential decay
    distribution of a $B\to V\ell^+\ell^-$ decay as defined in eq. (3.9)-(3.11)
    of arXiv:0811.1214.

    Input:
      - transversity_amps: dictionary containing the transverity amplitudes
      - q2: dilepton invariant mass squared $q_2$ in GeV$^2$
      - par: parameter dictionary
      - lep: lepton flavour, either 'e', 'mu', or 'tau'
    """
    ml = par[('mass',lep)]
    ApaL = transversity_amps['para_L']
    ApaR = transversity_amps['para_R']
    ApeL = transversity_amps['perp_L']
    ApeR = transversity_amps['perp_R']
    A0L = transversity_amps['0_L']
    A0R = transversity_amps['0_R']
    AS = transversity_amps['S']
    At = transversity_amps['t']
    J = {}
    J['1s'] = (1/(4 * q2) * ((-4 * ml**2 + 3 * q2) * abs(ApaL)**2
     + (-4 * ml**2 + 3 * q2) * abs(ApaR)**2 - 4 * ml**2 * abs(ApeL)**2 + 3 * q2 * abs(ApeL)**2 -
     4 * ml**2 * abs(ApeR)**2 + 3 * q2 * abs(ApeR)**2 +
     16 * ml**2 * np.real(ApaL * np.conj(ApaR)) +
     16 * ml**2 * np.real(ApeL * np.conj(ApeR))))
    J['1c'] = (abs(A0L)**2 + abs(A0R)**2 + (4 * ml**2 * abs(At)**2)/q2
     + beta_l(ml, q2)**2 * abs(AS)**2 + (8 * ml**2 * np.real(A0L * np.conj(A0R)))/q2)
    J['2s'] = (1/4 * beta_l(ml, q2)**2 * (abs(ApaL)**2 + abs(ApaR)**2 + abs(ApeL)**2 + abs(ApeR)**2))
    J['2c'] = (-beta_l(ml, q2)**2 * (abs(A0L)**2 + abs(A0R)**2))
    J[3] = (1/2 * beta_l(ml, q2)**2 * (-abs(ApaL)**2 - abs(ApaR)**2 + abs(ApeL)**2 + abs(ApeR)**2))
    J[4] = ((beta_l(ml, q2)**2 * (np.real(A0L * np.conj(ApaL)) + np.real(A0R * np.conj(ApaR))))/sqrt(2))
    J[5] = (sqrt(2) * beta_l(ml, q2) * (np.real(A0L * np.conj(ApeL)) - np.real(A0R * np.conj(ApeR)) -
     ml /sqrt(q2) * (np.real(ApaL * np.conj(AS)) + np.real(ApaR * np.conj(AS)))))
    J['6s'] = (2 * beta_l(ml, q2) * (np.real(ApaL * np.conj(ApeL)) - np.real(ApaR * np.conj(ApeR))))
    J['6c'] = (4 * ml/sqrt(q2) * beta_l(ml, q2) * (np.real(A0L * np.conj(AS)) + np.real(A0R * np.conj(AS))))
    J[7] = (sqrt(2) * beta_l(ml, q2) * (np.imag(A0L * np.conj(ApaL)) - np.imag(A0R * np.conj(ApaR)) +
     ml/sqrt(q2) * (np.imag(ApeL * np.conj(AS)) + np.imag(ApeR * np.conj(AS)))))
    J[8] = ((beta_l(ml, q2)**2 * (np.imag(A0L * np.conj(ApeL)) + np.imag(A0R * np.conj(ApeR))))/sqrt(2))
    J[9] = (-beta_l(ml, q2)**2 * (np.imag(ApaL * np.conj(ApeL)) + np.imag(ApaR * np.conj(ApeR))))
    return J

def angulardist_bar(transversity_amps_bar, q2, par, lep):
    J = angulardist(transversity_amps_bar, q2, par, lep)
    J[5] = -J[5]
    J['6s'] = -J['6s']
    J['6c'] = -J['6c']
    J[8] = -J[8]
    J[9] = -J[9]
    return J

def dGdq2(J):
    return 3/4. * (2 * J['1s'] + J['1c']) - 1/4. * (2 * J['2s'] + J['2c'])

def S_theory(J, J_bar, i):
    r"""CP-averaged angular observable $S_i$ in the theory convention."""
    dG     = dGdq2(J)
    dG_bar = dGdq2(J_bar)
    return (J[i] + J_bar[i])/(dG + dG_bar)

def A_theory(J, J_bar, i):
    r"""Angular CP asymmetry $A_i$ in the theory convention."""
    dG     = dGdq2(J)
    dG_bar = dGdq2(J_bar)
    return (J[i] - J_bar[i])/(dG + dG_bar)

def S_experiment(J, J_bar, i):
    r"""CP-averaged angular observable $S_i$ in the LHCb convention.

    See eq. (C.8) of arXiv:1506.03970v2.
    """
    if i in [4, '6s', '6c', 7, 9]:
        return -S_theory(J, J_bar, i)
    return S_theory(J, J_bar, i)

def A_experiment(J, J_bar, i):
    r"""Angular CP asymmetry $A_i$ in the LHCb convention.

    See eq. (C.8) of arXiv:1506.03970v2.
    """
    if i in [4, '6s', '6c', 7, 9]:
        return -A_theory(J, J_bar, i)
    return A_theory(J, J_bar, i)

def Pp_experiment(J, J_bar, i):
    r"""Observable $P'_i$ in the LHCb convention.

    See eq. (C.9) of arXiv:1506.03970v2.
    """
    Pp_to_S = {4: 4, 5: 5, 6: 7, 8: 8}
    if i not in Pp_to_S.keys():
        return ValueError("Observable P'_" + i + " not defined")
    denom = sqrt(FL(J, J_bar)*(1 - FL(J, J_bar)))
    return S_experiment(J, J_bar, Pp_to_S[i]) / denom

def AFB_experiment(J, J_bar):
    r"""Forward-backward asymmetry in the LHCb convention.

    See eq. (C.9) of arXiv:1506.03970v2.
    """
    return 3/4.*S_experiment(J, J_bar, '6s')

def AFB_theory(J, J_bar):
    """Forward-backward asymmetry in the original theory convention.
    """
    return 3/4.*S_theory(J, J_bar, '6s')

def FL(J, J_bar):
    r"""Longitudinal polarization fraction $F_L$"""
    return -S_theory(J, J_bar, '2c')

def FLhat(J, J_bar):
    r"""Modified longitudinal polarization fraction for vanishing lepton masses,
    $\hat F_L$.

    See eq. (32) of arXiv:1510.04239.
    """
    return -S_theory(J, J_bar, '1c')

def bvll_obs(function, q2, wc_obj, par, B, V, lep):
    scale = config['bdecays']['scale_bvll']
    wc = wctot_dict(wc_obj, 'df1_' + meson_quark[(B,V)], scale, par)
    a = transversity_amps(q2, wc, par, B, V, lep)
    J = angulardist(a, q2, par, lep)
    J_bar = angulardist_bar(a, q2, par, lep)
    return function(J, J_bar)

def bvll_dbrdq2(q2, wc_obj, par, B, V, lep):
    tauB = par[('lifetime',B)]
    fct = lambda J, J_bar: ( dGdq2(J) + dGdq2(J_bar) )/2.
    dGave = bvll_obs(fct, q2, wc_obj, par, B, V, lep)
    return tauB * dGave
