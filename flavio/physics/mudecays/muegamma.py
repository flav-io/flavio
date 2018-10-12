r"""Functions for lepton flavour violating $\tau\to \ell\gamma$ decays."""

import flavio
from flavio.physics.taudecays.taulgamma import BR_llgamma


def BR_muegamma(wc_obj, par):
    r"""Branching ratio of $\mu\to e\gamma$."""
    return BR_llgamma(wc_obj, par, 'mu', 'e')


_process_tex = r"\mu\to e\gamma"
_process_taxonomy = r'Process :: muon decays :: LFV decays :: $' + _process_tex + r"$"

_obs_name = "BR(mu->egamma)"
_obs = flavio.classes.Observable(_obs_name)
_obs.set_description(r"Branching ratio of $" + _process_tex + r"$")
_obs.tex = r"$\text{BR}(" + _process_tex + r")$"
_obs.add_taxonomy(_process_taxonomy)
flavio.classes.Prediction(_obs_name, BR_muegamma)
