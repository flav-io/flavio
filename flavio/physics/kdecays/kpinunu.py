r"""Observables in $K\to \pi\nu\bar\nu$ decays"""

import flavio
from flavio.classes import Observable, Prediction
from math import sqrt, pi

def kpinunu_amplitude(wc_obj, par, nu1, nu2):
    # This is the amplitude for $K\to\pi\nu\bar\nu$ (charged or neutral) in
    # some convenient normalization
    # top and charm CKM factors
    xi_t = flavio.physics.ckm.xi('t', 'ds')(par)
    xi_c = flavio.physics.ckm.xi('c', 'ds')(par)
    label = 'sd' + nu1 + nu2 # e.g. sdnuenue, sdnutaunumu
    scale = flavio.config['renormalization scale']['kdecays']
    # Wilson coefficients
    wc = wc_obj.get_wc(label, scale, par, nf_out=3)
    if nu1 == nu2: # add the SM contribution only if neutrino flavours coincide
        wc['CL_'+label] += flavio.physics.bdecays.wilsoncoefficients.CL_SM(par, scale)
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
    flavio.citations.register("Buras:2006gb")
    return 0.379 * (mc/1.3)**2.155 * (alpha_s/0.1187)**(-1.417)

def kappa_nu_tilde(par, charge):
    r"""$\kappa_\nu^{+,L}$ parameters in units of $1/|V_{us}|^8$ from arXiv:0705.2025"""
    flavio.citations.register("Mescia:2007kn")
    if charge == 'plus':
        kappa_tilde = par['kappa_plus_tilde']
    elif charge == 'long':
        kappa_tilde = par['kappa_L_tilde']
    else:
        raise ValueError("Argument `charge` must be 'plus' or 'long'")
    scale = flavio.config['renormalization scale']['kdecays']
    alpha_e_scale = flavio.physics.running.running.get_alpha(par, scale)['alpha_e']
    r_Vus = 1/0.225**8 # this is because kappa_nu is defined for a fixed value of Vus!
    r_alpha = alpha_e_scale**2 * 127.9**2 # 0705.2025 Eq. (11), convert alpha(MZ) to alpha(mu)
    r_s2w = 0.231**2 / par['s2w']**2 # 0705.2025 Eq. (11)
    return kappa_tilde * r_Vus * r_alpha * r_s2w

def br_kplus_pinunu(wc_obj, par, nu1, nu2):
    r"""Branching ratio of $K^+\pi^+\nu\bar\nu$ for fixed neutrino flavours"""
    amp = kpinunu_amplitude(wc_obj, par, nu1, nu2)
    Vus = abs(flavio.physics.ckm.get_ckm(par)[0,1])
    return abs(amp)**2 * kappa_nu_tilde(par, 'plus') / Vus**2 / 3.

def br_klong_pinunu(wc_obj, par, nu1, nu2):
    r"""Branching ratio of $K_L\pi^0\nu\bar\nu$ for fixed neutrino flavours"""
    amp = kpinunu_amplitude(wc_obj, par, nu1, nu2)
    Vus = abs(flavio.physics.ckm.get_ckm(par)[0,1])
    return amp.imag**2 * kappa_nu_tilde(par, 'long') / Vus**2 / 3.

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

_process_taxonomy = r'Process :: $s$ hadron decays :: FCNC decays :: $K\to \pi\nu\bar\nu$ :: '

_obs_name = "BR(K+->pinunu)"
_obs = flavio.classes.Observable(name=_obs_name)
_tex = r"K^+\to\pi^+\nu\bar\nu"
_obs.set_description(r"Branching ratio of $" + _tex + r"$")
_obs.tex = r"$\text{BR}(" + _tex + r")$"
_obs.add_taxonomy(_process_taxonomy + r'$' + _tex + r'$')
flavio.classes.Prediction(_obs_name, br_kplus_pinunu_summed)

_obs_name = "BR(KL->pinunu)"
_obs = flavio.classes.Observable(name=_obs_name)
_tex = r"K_L\to\pi^0\nu\bar\nu"
_obs.set_description(r"Branching ratio of $" + _tex + r"$")
_obs.tex = r"$\text{BR}(" + _tex + r")$"
_obs.add_taxonomy(_process_taxonomy + r'$' + _tex + r'$')
flavio.classes.Prediction(_obs_name, br_klong_pinunu_summed)
