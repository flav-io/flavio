r"""Functions for lepton flavour violating $\tau\to \ell\gamma$ decays."""

import flavio
from flavio.physics.taudecays.common import GammaFvf


# names of LFV sectors in WCxf
wcxf_sector_names = {('tau', 'mu'): 'mutau',
                     ('tau', 'e'): 'taue',
                     ('mu', 'e'): 'mue', }


def BR_llgamma(wc_obj, par, l1, l2):
    r"""Branching ratio of $\ell_1\to \ell_2\gamma$."""
    scale = flavio.config['renormalization scale'][l1 + 'decays']
    ll = wcxf_sector_names[l1, l2]
    wc = wc_obj.get_wc(ll, scale, par, nf_out=4)
    ml1 = par['m_' + l1]
    ml2 = par['m_' + l2]
    gTL = 2 * wc['Cgamma_' + l1 + l2].conjugate()
    gTR = 2 * wc['Cgamma_' + l2 + l1]
    return (par['tau_' + l1] * GammaFvf(M=ml1, mv=0, mf=ml2,
                                        gL=0, gR=0,
                                        gTL=gTL, gtTL=0,
                                        gTR=gTR, gtTR=0))


def BR_taulgamma(wc_obj, par, lep):
    r"""Branching ratio of $\tau\to \ell\gamma$."""
    return BR_llgamma(wc_obj, par, 'tau', lep)


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
