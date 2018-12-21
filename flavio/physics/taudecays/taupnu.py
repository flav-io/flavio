r"""Functions for $\tau\to P\nu$."""

import flavio
from flavio.physics.taudecays import common
from flavio.physics.bdecays.wilsoncoefficients import get_CVLSM
from math import sqrt
from flavio.physics.taudecays.taulnunu import GFeff


def _br_taupnu(wc_obj, par, P, lep):
    r"""Branching ratio of $\tau^+\to P^+\bar\nu_\ell$."""
    # CKM element
    scale = flavio.config['renormalization scale']['taudecays']
    if P == 'pi+':
        Vij = flavio.physics.ckm.get_ckm(par)[0, 0]  # Vud
        qqlnu = 'dutaunu' + lep
        mq1 = flavio.physics.running.running.get_md(par, scale)
        mq2 = flavio.physics.running.running.get_mu(par, scale)
    elif P == 'K+':
        Vij = flavio.physics.ckm.get_ckm(par)[0, 1]  # Vus
        qqlnu = 'sutaunu' + lep
        mq1 = flavio.physics.running.running.get_ms(par, scale)
        mq2 = flavio.physics.running.running.get_mu(par, scale)
    # Wilson coefficients
    wc = wc_obj.get_wc(qqlnu, scale, par, nf_out=4)
    # add SM contribution to Wilson coefficient
    if lep == 'tau':
        # for the SM contribution, need the Fermi constant with possible
        # NP effects in mu->enunu subtracted, not the measured one
        r_GF = GFeff(wc_obj, par) / par['GF']
        wc['CVL_' + qqlnu] += get_CVLSM(par, scale, nf=4) * r_GF
    mtau = par['m_tau']
    mP = par['m_' + P]
    rWC = ((wc['CVL_' + qqlnu] - wc['CVR_' + qqlnu])
           + mP**2 / mtau / (mq1 + mq2) * (wc['CSR_' + qqlnu] - wc['CSL_' + qqlnu]))
    gR = -sqrt(2) * par['GF'] * Vij * rWC * par['f_' + P] * par['m_tau']
    return par['tau_tau'] * common.GammaFsf(mtau, mP, 0, 0, gR)


def br_taupnu(wc_obj, par, P):
    r"""Branching ratio of $\tau^+\to P^+\bar\nu_\ell$, summing over the
    neutrino flavour."""
    return sum([_br_taupnu(wc_obj, par, P, lep) for lep in ['e', 'mu', 'tau']])


# function returning function needed for prediction instance
def br_taupnu_fct(P):
    def f(wc_obj, par):
        return br_taupnu(wc_obj, par, P)
    return f

# Observable and Prediction instances

_had = {'pi+': r'\pi^+', 'K+': r'K^+',}
_shortname = {'pi+': 'pi', 'K+': 'K'}

for P in _had:
    _obs_name = "BR(tau->" + _shortname[P]+"nu)"
    _obs = flavio.classes.Observable(_obs_name)
    _process_tex = r"\tau^+\to " + _had[P] + r"\bar\nu"
    _process_taxonomy = r'Process :: $\tau$ lepton decays :: Hadronic tree-level decays :: $\tau\to V\ell$ :: $' + _process_tex + r"$"
    _obs.add_taxonomy(_process_taxonomy)
    _obs.set_description(r"Branching ratio of $" + _process_tex + r"$")
    _obs.tex = r"$\text{BR}(" + _process_tex + r")$"
    flavio.classes.Prediction(_obs_name, br_taupnu_fct(P))
