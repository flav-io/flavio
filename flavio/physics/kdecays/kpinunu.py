r"""Observables in $K\to \pi\nu\bar\nu$ decays"""

import flavio
from flavio.classes import Observable, Prediction
from math import sqrt, pi

def kpinunu_amplitude(wc_obj, par, nu1, nu2):
    # This is the amplitude for $K\to\pi\nu\bar\nu$ (charged or neutral) in
    # some convenient normalization
    # top and charm CKM factors
    xi_t = flavio.physics.ckm.xi('t', 'sd')(par)
    xi_c = flavio.physics.ckm.xi('c', 'sd')(par)
    label = 'sd' + nu1 + nu2 # e.g. sdnuenue, sdnutaunumu
    scale = flavio.config['renormalization scale']['kdecays']
    # Wilson coefficients
    wc = wc_obj.get_wc(label, scale, par)
    if nu1 == nu2: # add the SM contribution only if neutrino flavours coincide
        wc['CL_'+label] += flavio.physics.bdecays.wilsoncoefficients.CL_SM(par)
    s2w = par['s2w']
    X = -( wc['CL_'+label] + wc['CR_'+label] ) * s2w
    Vus = abs(flavio.physics.ckm.get_ckm(par)[0,1])
    PcX = kpinunu_charm(par)
    deltaPcu = par['deltaPcu']
    amp = xi_t * X # top contribution
    if nu1 == nu2: # add the charm contribution if neutrino flavours coincide
        amp += xi_c * (PcX + deltaPcu)  * Vus**4
    return amp

def kpinunu_charm(par):
    r"""Charm contribution to the $K\to\pi\nu\bar\nu$ amplitude conventionally
    denoted as $P_c(X)$"""
    mc = par['m_c']
    alpha_s = par['alpha_s']
    # approximate formula for NNLO perturbative result: (14) of hep-ph/0603079
    return 0.379 * (mc/1.3)**2.155 * (alpha_s/0.1187)**(-1.417)

def br_kplus_pinunu(wc_obj, par, nu1, nu2):
    r"""Branching ratio of $K^+\pi^+\nu\bar\nu$ for fixed neutrino flavours"""
    amp = kpinunu_amplitude(wc_obj, par, nu1, nu2)
    Vus_tilde = 0.225 #  this is because kappa_plus is defined for a fixed value of Vus!
    Vus = abs(flavio.physics.ckm.get_ckm(par)[0,1])
    return abs(amp)**2 * par['kappa_plus_tilde'] / Vus_tilde**8 / Vus**2 / 3.

def br_klong_pinunu(wc_obj, par, nu1, nu2):
    r"""Branching ratio of $K_L\pi^0\nu\bar\nu$ for fixed neutrino flavours"""
    amp = kpinunu_amplitude(wc_obj, par, nu1, nu2)
    Vus_tilde = 0.225 #  this is because kappa_plus is defined for a fixed value of Vus!
    Vus = abs(flavio.physics.ckm.get_ckm(par)[0,1])
    return amp.imag**2 * par['kappa_L_tilde'] / Vus_tilde**8 / Vus**2 / 3.

def br_kplus_pinunu_summed(wc_obj, par):
    r"""Branching ratio of $K^+\pi^+\nu\bar\nu$ summed over neutrino flavours"""
    f = ['nue', 'numu', 'nutau']
    brs = [ br_kplus_pinunu(wc_obj, par, nu1, nu2) for nu1 in f for nu2 in f ]
    return sum(brs)

def br_klong_pinunu_summed(wc_obj, par):
    r"""Branching ratio of $K_L\pi^0\nu\bar\nu$ summed over neutrino flavours"""
    f = ['nue', 'numu', 'nutau']
    brs = [ br_klong_pinunu(wc_obj, par, nu1, nu2) for nu1 in f for nu2 in f ]
    return sum(brs)

# Observable and Prediction instances

_obs_name = "BR(K+->pinunu)"
_obs = flavio.classes.Observable(name=_obs_name)
_tex = r"K^+\to\pi^+\nu\bar\nu"
_obs.set_description(r"Branching ratio of $" + _tex + r"$")
_obs.tex = r"$\text{BR}(" + _tex + r")$"
flavio.classes.Prediction(_obs_name, br_kplus_pinunu_summed)

_obs_name = "BR(KL->pinunu)"
_obs = flavio.classes.Observable(name=_obs_name)
_tex = r"K_L\to\pi^0\nu\bar\nu"
_obs.set_description(r"Branching ratio of $" + _tex + r"$")
_obs.tex = r"$\text{BR}(" + _tex + r")$"
flavio.classes.Prediction(_obs_name, br_klong_pinunu_summed)
