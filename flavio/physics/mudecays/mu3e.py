r"""Functions for the lepton flavour violating $\mu\to 3e$ decay."""

import flavio
from math import pi, sqrt

from flavio.physics.taudecays.tau3l import _BR_tau3mu


def BR_mu3e(wc_obj, par):
    r"""Branching ratio of $\mu^-\to e^-e^+e^-$."""
    scale = flavio.config['renormalization scale']['mudecays']
    wc = wc_obj.get_wc('mue', scale, par, nf_out=4)
    alpha = flavio.physics.running.running.get_alpha_e(par, scale, nf_out=3)
    e = sqrt(4 * pi * alpha)
    # cf. (22, 23) of hep-ph/0404211
    pre_br = 1 / (8 * par['GF']**2)
    mmu = par['m_mu']
    me = par['m_e']
    pre_wc_1 = 1 / e / mmu
    pre_wc_2 = 1
    DLg = pre_wc_1 * wc['Cgamma_mue']
    DRg = pre_wc_1 * wc['Cgamma_emu'].conjugate()
    FLL = pre_wc_2 * wc['CVLL_eemue']
    FLR = pre_wc_2 * wc['CVLR_eemue']
    FRL = pre_wc_2 * wc['CVLR_mueee']
    FRR = pre_wc_2 * wc['CVRR_eemue']
    SRR = pre_wc_2 * wc['CSRR_eemue']
    SLL = pre_wc_2 * wc['CSRR_eeemu'].conjugate()
    br_wc = _BR_tau3mu(mmu, me, FLL, FLR, FRL, FRR, DLg, DRg, SRR, SLL)
    return pre_br * br_wc


_process_tex = r"\mu^-\to e^-e^+e^-"
_process_taxonomy = r'Process :: muon decays :: LFV decays :: $' + _process_tex + r"$"

_obs_name = "BR(mu->eee)"
_obs = flavio.classes.Observable(_obs_name)
_obs.set_description(r"Branching ratio of $" + _process_tex + r"$")
_obs.tex = r"$\text{BR}(" + _process_tex + r")$"
_obs.add_taxonomy(_process_taxonomy)
flavio.classes.Prediction(_obs_name, BR_mu3e)
