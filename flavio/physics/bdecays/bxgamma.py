r"""Functions for inclusive $B\to X_q\gamma$ decays."""

import flavio
import numpy as np
from math import pi
from flavio.classes import Observable, Prediction

# NLO and NNLO coefficients K_{ij}^{(1,2)} (see e.g. eq. (3.1) of hep-ph/0609241)
# Mikolaj Misiak, private communication
# see arXiv:1503.01789 and references therein
ka1 = np.array([[0.00289054,-0.0173432,-0.000837297,0.00013955,-0.0158512,-0.101529,0.0913267,-0.00166875],
                [-0.0173432,0.104059,0.00502378,-0.000837297,0.0951075,0.609174,-0.54796,0.0100125],
                [-0.000837297,0.00502378,0.028828,-0.00480467,0.318696,-0.0282896,8.32249,-0.0187533],
                [0.00013955,-0.000837297,-0.00480467,0.000800779,-0.053116,0.00471493,-1.61065,0.00312556],
                [-0.0158512,0.0951075,0.318696,-0.053116,3.54919,-0.0840081,122.732,-0.258893],
                [-0.101529,0.609174,-0.0282896,0.00471493,-0.0840081,1.80697,19.5562,0.108369],
                [0.0913267,-0.54796,8.32249,-1.61065,122.732,19.5562,5.61819,-0.50728],
                [-0.00166875,0.0100125,-0.0187533,0.00312556,-0.258893,0.108369,-0.50728,0.452362]])
ka2 = np.array([[0.114381,-0.686288,0,0,0,0,9.11217,0.215745],
                [-0.686288,4.11773,0,0,0,0,-8.85918,-1.29447],
                [0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
                [9.11217,-8.85918,0,0,0,0,-37.3173,-13.4112],
                [0.215745,-1.29447,0,0,0,0,-13.4112,22.3184]])

def BRBXgamma(wc_obj, par, q, E0):
    r"""Branching ratio of $B\to X_q\gamma$ ($q=s$ or $d$) normalized to
    $B\to X_c\ell\nu$ taken from experiment. `E0` is the photon energy
    cutoff $E_0$ in GeV
    (currently works only for `E0=1.6`).
    See arXiv:1503.01789 and references therein."""
    scale = flavio.config['renormalization scale']['bxgamma']
    alphaem = 1/137.035999139 # this is alpha_e(0), a constant for our purposes
    bq = 'b' + q
    xi_t = flavio.physics.ckm.xi('t',bq)(par)
    Vcb = flavio.physics.ckm.get_ckm(par)[1,2]
    C = par['C_BXlnu']
    BRSL = par['BR(B->Xcenu)_exp']
    # these are the b->qee Wilson coefficients - they contain the b->qgamma ones as a subset
    wc = flavio.physics.bdecays.wilsoncoefficients.wctot_dict(wc_obj, bq+'ee', scale, par, nf_out=5)
    PE0 = PE0_BXgamma(wc, par, q, E0, CPodd=0)
    # uncertainty due to higher order contributions, interpolation, and nonperturbative effects
    P_uncertainty = 0.1249 * par['delta_BX'+q+'gamma'] # PE0_NLO * delta
    # central value of non-perturbative correction (M. Misiak, private communication)
    P_nonpert = 0.00381745
    # eq. (2.1) of hep-ph/0609241
    return BRSL * abs(xi_t)**2/abs(Vcb)**2 * 6*alphaem/pi/C * (PE0 + P_nonpert + P_uncertainty)

def PE0_BXgamma(wc, par, q, E0, CPodd):
    r"""Branching ratio of $B\to X_q\gamma$ without the prefactor.
    At leading order in the SM, this function is equal to
    $(C_7^{\mathrm{eff}})^2$."""
    bq = 'b' + q
    if E0 != 1.6:
        raise ValueError("BR(B->Xqgamma) is not implemented for E0 different from 1.6 GeV")
    P0 = abs(wc['C7eff_'+bq])**2 + abs(wc['C7effp_'+bq])**2
    scale = flavio.config['renormalization scale']['bxgamma']
    alphas = flavio.physics.running.running.get_alpha(par, scale, nf_out=5)['alpha_s']
    at = alphas/4./pi
    coeffs = ['C1_'+bq, 'C2_'+bq, 'C3_'+bq, 'C4_'+bq, 'C5_'+bq, 'C6_'+bq, 'C7eff_'+bq, 'C8eff_'+bq]
    coeffs_p = ['C1p_'+bq, 'C2p_'+bq, 'C3p_'+bq, 'C4p_'+bq, 'C5p_'+bq, 'C6p_'+bq, 'C7effp_'+bq, 'C8effp_'+bq]
    wc1_8 = np.array([wc[name] for name in coeffs])
    wc1p_8p = np.array([wc[name] for name in coeffs_p])
    P1 = at    * np.dot(wc1_8, np.dot(ka1, wc1_8.conj())).real
    P1_p = at    * np.dot(wc1p_8p, np.dot(ka1, wc1p_8p.conj())).real
    P2 = at**2 * np.dot(wc1_8, np.dot(ka2, wc1_8.conj())).real
    P2_p = at**2 * np.dot(wc1p_8p, np.dot(ka2, wc1p_8p.conj())).real
    Prest = P1 + P2 + P1_p + P2_p
    r_u = flavio.physics.ckm.xi('u',bq)(par)/flavio.physics.ckm.xi('t',bq)(par)
    PVub = -0.0296854 * r_u.real + CPodd * 0.0596211 * r_u.imag + 0.123411 * abs(r_u)**2
    return P0 + Prest + PVub


_obs_name = "BR(B->Xsgamma)"
_obs = Observable(_obs_name)
_obs.set_description(r"CP-averaged branching ratio of $B\to X_s\gamma$ for $E_\gamma>1.6$ GeV")
_obs.tex = r"$\text{BR}(B\to X_s\gamma)$"
Prediction(_obs_name, lambda wc_obj, par: BRBXgamma(wc_obj, par, 's', 1.6))

_obs_name = "BR(B->Xdgamma)"
_obs = Observable(_obs_name)
_obs.set_description(r"CP-averaged branching ratio of $B\to X_d\gamma$ for $E_\gamma>1.6$ GeV")
_obs.tex = r"$\text{BR}(B\to X_d\gamma)$"
Prediction(_obs_name, lambda wc_obj, par: BRBXgamma(wc_obj, par, 'd', 1.6))
