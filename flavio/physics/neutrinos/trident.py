r"""Functions for neutrino trident production, $\nu N\to \nu\ell^+\ell^- N$"""

import flavio


def R_trident(wc_obj, par):
    r"""Neutrino trident production cross section normalized to the SM value.

    The process is $\nu_\mu N\to \nu_\ell\mu^+\mu^- \nu N$."""
    scale = 1  # doesn't matter at all since coefficients don't run!
    wc = wc_obj.get_wc(sector='ffnunu', scale=scale, par=par, eft='WET-3', basis='flavio')
    # CV is defined as the Wilson coefficient of
    # - 4 GF / sqrt(2) (\bar\nu_L \gamma^\mu \nu_L)(\bar\mu \gamma^\mu \mu)
    # CA is defined as the Wilson coefficient of
    # - 4 GF / sqrt(2) (\bar\nu_L \gamma^\mu \nu_L)(\bar\mu \gamma^\mu \gamma_5 \mu)
    # SM contribution (W + Z exchange)
    CV_SM = 1 / 2 + 2 * par['s2w']
    CA_SM = 1 / 2
    # NP contributions: final state nu_mu ...
    CV_NP_m = -wc['CVLR_numunumumumu'] - wc['CVLL_numunumumumu']
    CA_NP_m = -wc['CVLL_numunumumumu'] + wc['CVLR_numunumumumu']
    # nu_e
    CV_NP_e = -wc['CVLL_nuenumumumu'] - wc['CVLR_nuenumumumu']
    CA_NP_e = -wc['CVLL_nuenumumumu'] + wc['CVLR_nuenumumumu']
    # nu_tau
    CV_NP_t = -wc['CVLL_numunutaumumu'] - wc['CVLR_numunutaumumu']
    CA_NP_t = -wc['CVLL_numunutaumumu'] + wc['CVLR_numunutaumumu']
    # coherent sum for final state nu_mu
    CA_m = CA_SM + CA_NP_m
    CV_m = CV_SM + CV_NP_m
    sigma_SM = abs(CA_SM)**2 + abs(CV_SM)**2
    # SM-NP interfering only for final-state nu_mu
    sigma_tot = abs(CA_m)**2 + abs(CV_m)**2 + abs(CA_NP_e)**2 + abs(CV_NP_e)**2 + abs(CA_NP_t)**2 + abs(CV_NP_t)**2
    return sigma_tot / sigma_SM


_process_tex = r"$\nu_\mu N\to \nu_\ell\mu^+\mu^-N$"
_process_taxonomy = r'Process :: neutrino physics :: scattering cross sections :: ' + _process_tex

_obs_name = "R_trident"
_obs = flavio.classes.Observable(_obs_name)
_obs.set_description(r"Cross section of neutrino trident production, {}, normalized to the SM".format(_process_tex))
_obs.tex = r"$\sigma_\text{trident}/\sigma_\text{trident}^\text{SM}$"
_obs.add_taxonomy(_process_taxonomy)
flavio.classes.Prediction(_obs_name, R_trident)
