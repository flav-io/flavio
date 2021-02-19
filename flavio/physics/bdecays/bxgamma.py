r"""Functions for inclusive $B\to X_q\gamma$ decays."""

import flavio
import numpy as np
from math import pi
from flavio.classes import Observable, Prediction

# NLO and NNLO coefficients K_{ij}^{(1,2)} (see e.g. eq. (3.1) of hep-ph/0609241)
# (real parts)
# Mikolaj Misiak, private communication
# see arXiv:1503.01789 and references therein
ka1_r = np.array([[0.00289054,-0.0173432,-0.000837297,0.00013955,-0.0158512,-0.101529,0.0913267,-0.00166875],
                [-0.0173432,0.104059,0.00502378,-0.000837297,0.0951075,0.609174,-0.54796,0.0100125],
                [-0.000837297,0.00502378,0.028828,-0.00480467,0.318696,-0.0282896,8.32249,-0.0187533],
                [0.00013955,-0.000837297,-0.00480467,0.000800779,-0.053116,0.00471493,-1.61065,0.00312556],
                [-0.0158512,0.0951075,0.318696,-0.053116,3.54919,-0.0840081,122.732,-0.258893],
                [-0.101529,0.609174,-0.0282896,0.00471493,-0.0840081,1.80697,19.5562,0.108369],
                [0.0913267,-0.54796,8.32249,-1.61065,122.732,19.5562,5.61819,-0.50728],
                [-0.00166875,0.0100125,-0.0187533,0.00312556,-0.258893,0.108369,-0.50728,0.452362]])
ka2_r = np.array([[0.114381,-0.686288,0,0,0,0,9.11217,0.215745],
                [-0.686288,4.11773,0,0,0,0,-8.85918,-1.29447],
                [0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
                [9.11217,-8.85918,0,0,0,0,-37.3173,-13.4112],
                [0.215745,-1.29447,0,0,0,0,-13.4112,22.3184]])

# imaginary part of K_{ij}^{(1)}
# Ayan Paul/HEPfit, private communication
ka1_i = np.array([[0, 0, -0.007562956672, 0.001260492779, -0.09028834311, 0.0163641251, 0.1092317551, 0.006116280995],
                  [0, 0, 0.04537774003, -0.007562956672, 0.5417300587, -0.0981847506, -0.6553905309, -0.03669768597],
                  [0.007562956672, -0.04537774003, 0, 0, 0, 0.2722664402, 1.085982646, 0],
                  [-0.001260492779, 0.007562956672, 0, 0, 0, -0.04537774003, -1.221446177, 0],
                  [0.09028834311, -0.5417300587, 0, 0, 0, 3.250380352, 17.37572233, 0],
                  [-0.0163641251, 0.0981847506, -0.2722664402, 0.04537774003, -3.250380352, 0, -16.57524067, -0.2201548215],
                  [-0.1092317551, 0.6553905309, -1.085982646, 1.221446177, -17.37572233, 16.57524067, 0, 2.792526803],
                  [-0.006116280995, 0.03669768597, 0, 0, 0, 0.2201548215, -2.792526803, 0]])

def BRBXgamma(wc_obj, par, q, E0):
    r"""Branching ratio of $B\to X_q\gamma$ ($q=s$ or $d$) normalized to
    $B\to X_c\ell\nu$ taken from experiment. `E0` is the photon energy
    cutoff $E_0$ in GeV
    (currently works only for `E0=1.6`).
    See arXiv:1503.01789 and references therein."""
    flavio.citations.register("Misiak:2015xwa")
    scale = flavio.config['renormalization scale']['bxgamma']
    alphaem = 1/137.035999139 # this is alpha_e(0), a constant for our purposes
    bq = 'b' + q
    xi_t = flavio.physics.ckm.xi('t',bq)(par)
    Vcb = flavio.physics.ckm.get_ckm(par)[1,2]
    C = par['C_BXlnu']
    BRSL = par['BR(B->Xcenu)_exp']
    # these are the b->qee Wilson coefficients - they contain the b->qgamma ones as a subset
    wc = flavio.physics.bdecays.wilsoncoefficients.wctot_dict(wc_obj, bq+'ee', scale, par, nf_out=5)
    PE0 = PE0_BR_BXgamma(wc, par, q, E0)
    # uncertainty due to higher order contributions, interpolation, and nonperturbative effects
    P_uncertainty = 0.1249 * par['delta_BX'+q+'gamma'] # PE0_NLO * delta
    # central value of non-perturbative correction (M. Misiak, private communication)
    P_nonpert = 0.00381745
    # eq. (2.1) of hep-ph/0609241
    flavio.citations.register("Misiak:2006ab")
    return BRSL * abs(xi_t)**2/abs(Vcb)**2 * 6*alphaem/pi/C * (PE0 + P_nonpert + P_uncertainty)

def PE0_BR_BXgamma(wc, par, q, E0):
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
    P1 = at    * np.dot(wc1_8, np.dot(ka1_r, wc1_8.conj())).real
    P1_p = at    * np.dot(wc1p_8p, np.dot(ka1_r, wc1p_8p.conj())).real
    P2 = at**2 * np.dot(wc1_8, np.dot(ka2_r, wc1_8.conj())).real
    P2_p = at**2 * np.dot(wc1p_8p, np.dot(ka2_r, wc1p_8p.conj())).real
    Prest = P1 + P2 + P1_p + P2_p
    r_u = flavio.physics.ckm.xi('u',bq)(par)/flavio.physics.ckm.xi('t',bq)(par)
    PVub = -0.0296854 * r_u.real + 0.123411 * abs(r_u)**2
    return P0 + Prest + PVub

def PE0_ACP_BXgamma(wc, par, q, E0):
    r"""Numerator of the direct CP asymmetry of $B\to X_q\gamma$ without
    the prefactor. This is the CP asymmetric analogue of PE0_BR_BXgamma."""
    bq = 'b' + q
    if E0 != 1.6:
        raise ValueError("ACP(B->Xgamma) is not implemented for E0 different from 1.6 GeV")
    scale = flavio.config['renormalization scale']['bxgamma']
    alphas = flavio.physics.running.running.get_alpha(par, scale, nf_out=5)['alpha_s']
    at = alphas/4./pi
    coeffs = ['C1_'+bq, 'C2_'+bq, 'C3_'+bq, 'C4_'+bq, 'C5_'+bq, 'C6_'+bq, 'C7eff_'+bq, 'C8eff_'+bq]
    coeffs_p = ['C1p_'+bq, 'C2p_'+bq, 'C3p_'+bq, 'C4p_'+bq, 'C5p_'+bq, 'C6p_'+bq, 'C7effp_'+bq, 'C8effp_'+bq]
    wc1_8 = np.array([wc[name] for name in coeffs])
    wc1p_8p = np.array([wc[name] for name in coeffs_p])
    P1 = at    * np.dot(wc1_8, np.dot(ka1_i, wc1_8.conj())).imag
    P1_p = at    * np.dot(wc1p_8p, np.dot(ka1_i, wc1p_8p.conj())).imag
    r_u = flavio.physics.ckm.xi('u',bq)(par)/flavio.physics.ckm.xi('t',bq)(par)
    PVub = 0.0596211 * r_u.imag
    return P1 + P1_p + PVub

def ACPBXgamma(wc_obj, par, E0):
    r"""Direct CP asymmetry of $B\to X_{s+d}\gamma$. `E0` is the photon energy
    cutoff $E_0$ in GeV (currently works only for `E0=1.6`)."""
    scale = flavio.config['renormalization scale']['bxgamma']
    # these are the b->qee Wilson coefficients - they contain the b->qgamma ones as a subset
    xi_t_d = flavio.physics.ckm.xi('t', 'bd')(par)
    xi_t_s = flavio.physics.ckm.xi('t', 'bs')(par)
    wc_d = flavio.physics.bdecays.wilsoncoefficients.wctot_dict(wc_obj, 'bdee', scale, par, nf_out=5)
    wc_s = flavio.physics.bdecays.wilsoncoefficients.wctot_dict(wc_obj, 'bsee', scale, par, nf_out=5)
    br_d = abs(xi_t_d)**2 * PE0_BR_BXgamma(wc_d, par, 'd', E0)
    br_s = abs(xi_t_s)**2 * PE0_BR_BXgamma(wc_s, par, 's', E0)
    as_d = abs(xi_t_d)**2 * PE0_ACP_BXgamma(wc_d, par, 'd', E0)
    as_s = abs(xi_t_s)**2 * PE0_ACP_BXgamma(wc_s, par, 's', E0)
    # return (as_s)/(br_s + br_d)
    return (as_s + as_d)/(br_s + br_d)

_process_taxonomy = r'Process :: $b$ hadron decays :: FCNC decays :: $B\to X\gamma$ :: '

_obs_name = "BR(B->Xsgamma)"
_obs = Observable(_obs_name)
_obs.set_description(r"CP-averaged branching ratio of $B\to X_s\gamma$ for $E_\gamma>1.6$ GeV")
_obs.tex = r"$\text{BR}(B\to X_s\gamma)$"
_obs.add_taxonomy(_process_taxonomy + r"$B\to X_s\gamma$")
Prediction(_obs_name, lambda wc_obj, par: BRBXgamma(wc_obj, par, 's', 1.6))

_obs_name = "BR(B->Xdgamma)"
_obs = Observable(_obs_name)
_obs.set_description(r"CP-averaged branching ratio of $B\to X_d\gamma$ for $E_\gamma>1.6$ GeV")
_obs.tex = r"$\text{BR}(B\to X_d\gamma)$"
_obs.add_taxonomy(_process_taxonomy + r"$B\to X_d\gamma$")
Prediction(_obs_name, lambda wc_obj, par: BRBXgamma(wc_obj, par, 'd', 1.6))

_obs_name = "ACP(B->Xgamma)"
_obs = Observable(_obs_name)
_obs.set_description(r"Direct CP asymmetry in $B\to X_{s+d}\gamma$ for $E_\gamma>1.6$ GeV")
_obs.tex = r"$A_\text{CP}(B\to X_{s+d}\gamma)$"
_obs.add_taxonomy(_process_taxonomy + r"$B\to X_{s+d}\gamma$")
Prediction(_obs_name, lambda wc_obj, par: ACPBXgamma(wc_obj, par, 1.6))
