r"""Functions for lepton flavour violating $\tau\to \ell\gamma$ decays."""

import flavio
from math import pi


# names of LFV sectors in WCxf
wcxf_sector_names = {('tau', 'mu'): 'mutau',
                     ('tau', 'e'): 'taue',
                     ('mu', 'e'): 'mue', }


def BR_llgamma(wc_obj, par, scale, l1, l2):
    r"""Branching ratio of $\ell_1\to \ell_2\gamma$."""
    scale = flavio.config['renormalization scale']['taudecays']
    ll = wcxf_sector_names[l1, l2]
    wc = wc_obj.get_wc(ll, scale, par, nf_out=4)
    alpha = flavio.physics.running.running.get_alpha_e(par, scale, nf_out=4)
    # cf. (18) of hep-ph/0404211
    pre = 6 * alpha / pi
    C7 = wc['Cgamma_' + l1 + l2]
    C7p = wc['Cgamma_' + l2 + l1].conjugate()
    if l1 == 'tau':
        BR_SL = par['BR(tau->{}nunu)'.format(l2)]
    else:
        BR_SL = 1  # BR(mu->enunu) = 1
    return pre * (abs(C7)**2 + abs(C7p)**2) * BR_SL


def BR_taulgamma(wc_obj, par, lep):
    r"""Branching ratio of $\tau\to \ell\gamma$."""
    scale = flavio.config['renormalization scale']['taudecays']
    return BR_llgamma(wc_obj, par, scale, 'tau', lep)


# function returning function needed for prediction instance
def br_taulgamma_fct(lep):
    def f(wc_obj, par):
        return BR_taulgamma(wc_obj, par, lep)
    return f


# Observable and Prediction instances
_tex = {'e': 'e', 'mu': r'\mu'}

for lep in _tex:
    _process_tex = r"\tau\to " + _tex[lep] + r"\gamma"
    _process_taxonomy = r'Process :: $\tau$ lepton decays :: LFV decays :: $\tau\to \ell\gamma$ :: $' + _process_tex + r"$"

    _obs_name = "BR(tau->" + lep + "gamma)"
    _obs = flavio.classes.Observable(_obs_name)
    _obs.set_description(r"Branching ratio of $" + _process_tex + r"$")
    _obs.tex = r"$\text{BR}(" + _process_tex + r")$"
    _obs.add_taxonomy(_process_taxonomy)
    flavio.classes.Prediction(_obs_name, br_taulgamma_fct(lep))
