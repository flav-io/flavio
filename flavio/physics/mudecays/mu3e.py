r"""Functions for the lepton flavour violating $\mu\to 3e$ decay."""

import flavio
from math import pi, sqrt

from flavio.physics.taudecays.tau3l import _BR_tau3mu, wc_eff


def BR_mu3e(wc_obj, par):
    r"""Branching ratio of $\mu^-\to e^-e^+e^-$."""
    scale = flavio.config['renormalization scale']['mudecays']
    wceff = wc_eff(wc_obj, par, scale, 'mu', 'e', 'e', 'e', nf_out=4)
    # cf. (22, 23) of hep-ph/0404211
    pre_br = 1 / (8 * par['GF']**2)
    br_wc = _BR_tau3mu(par['m_mu'], par['m_e'], wceff)
    return pre_br * br_wc


_process_tex = r"\mu^-\to e^-e^+e^-"
_process_taxonomy = r'Process :: muon decays :: LFV decays :: $' + _process_tex + r"$"

_obs_name = "BR(mu->eee)"
_obs = flavio.classes.Observable(_obs_name)
_obs.set_description(r"Branching ratio of $" + _process_tex + r"$")
_obs.tex = r"$\text{BR}(" + _process_tex + r")$"
_obs.add_taxonomy(_process_taxonomy)
flavio.classes.Prediction(_obs_name, BR_mu3e)
