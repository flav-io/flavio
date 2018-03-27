r"""Functions for lepton flavour violating $\tau\to \ell_1\ell_2\ell_2$ decays."""

import flavio
from math import pi, log, sqrt


# names of LFV sectors in WCxf
wcxf_sector_names = {('tau', 'mu'): 'mutau',
                     ('tau', 'e'): 'taue',
                     ('mu', 'e'): 'mue', }


def _BR_taumuee(mtau, me, FLL, FLR, FRL, FRR, e2DLg, e2DRg, SRR, SLL):
    # (22) of hep-ph/0404211
    # FIXME scalar contributions missing!
    return (abs(FLL)**2 + abs(FLR)**2 + abs(FRL)**2 + abs(FRR)**2
            + 4 * (e2DLg * (FLL + FLR).conjugate()
                          + e2DRg * (FRL + FRR).conjugate()).real
            + 8 * (abs(e2DLg)**2 + abs(e2DRg)**2)
            * (log(mtau**2 / me**2) - 3))


def _BR_tau3mu(mtau, mmu, FLL, FLR, FRL, FRR, e2DLg, e2DRg, SRR, SLL):
    # (23) of hep-ph/0404211
    # (117) of hep-ph/9909265
    return (2 * abs(FLL)**2 + abs(FLR)**2 + abs(FRL)**2 + 2 * abs(FRR)**2
            + 1 / 8 * (abs(SLL)**2 + abs(SRR)**2)
            + 4 * (e2DLg * (2 * FLL + FLR).conjugate()
                          + e2DRg * (FRL + 2 * FRR).conjugate()).real
            + 8 * (abs(e2DLg)**2 + abs(e2DRg)**2)
            * (log(mtau**2 / mmu**2) - 11 / 4))


def BR_taumull(wc_obj, par, lep):
    r"""Branching ratio of $\tau^-\to\mu^-\ell^+\ell^-$."""
    scale = flavio.config['renormalization scale']['taudecays']
    sector = wcxf_sector_names['tau', 'mu']
    wc = wc_obj.get_wc(sector, scale, par, nf_out=4)
    alpha = flavio.physics.running.running.get_alpha_e(par, scale, nf_out=4)
    e = sqrt(4 * pi * alpha)
    # cf. (22, 23) of hep-ph/0404211
    pre_br = par['BR(tau->mununu)'] / (8 * par['GF']**2)
    mtau = par['m_tau']
    pre_wc_1 = 1 / e / mtau
    pre_wc_2 = 1
    e2DLg = e**2 * pre_wc_1 * wc['Cgamma_taumu']
    e2DRg = e**2 * pre_wc_1 * wc['Cgamma_mutau'].conjugate()
    FLL = pre_wc_2 * wc['CVLL_{}taumu'.format(2 * lep)]
    FLR = pre_wc_2 * wc['CVLR_{}taumu'.format(2 * lep)]
    FRL = pre_wc_2 * wc['CVLR_taumu{}'.format(2 * lep)]
    FRR = pre_wc_2 * wc['CVRR_{}taumu'.format(2 * lep)]
    SRR = pre_wc_2 * wc['CSRR_{}taumu'.format(2 * lep)]
    SLL = pre_wc_2 * wc['CSRR_{}mutau'.format(2 * lep)].conjugate()
    if lep == 'e':
        me = par['m_e']
        br_wc = _BR_taumuee(mtau, me, FLL, FLR, FRL, FRR, e2DLg, e2DRg, SRR, SLL)
    elif lep == 'mu':
        mmu = par['m_mu']
        br_wc = _BR_tau3mu(mtau, mmu, FLL, FLR, FRL, FRR, e2DLg, e2DRg, SRR, SLL)
    return pre_br * br_wc


# function returning function needed for prediction instance
def br_taumull_fct(lep):
    def f(wc_obj, par):
        return BR_taumull(wc_obj, par, lep)
    return f


# Observable and Prediction instances
_tex = {'e': 'e', 'mu': r'\mu'}

for lep in _tex:
    _process_tex = r"\tau^-\to \mu^-" + _tex[lep] + r"^+" + _tex[lep] + r"^-"
    _process_taxonomy = r'Process :: $\tau$ lepton decays :: LFV decays :: $\tau\to \ell^\prime\ell\ell$ :: $' + _process_tex + r"$"

    _obs_name = "BR(tau->mu" + 2 * lep + ")"
    _obs = flavio.classes.Observable(_obs_name)
    _obs.set_description(r"Branching ratio of $" + _process_tex + r"$")
    _obs.tex = r"$\text{BR}(" + _process_tex + r")$"
    _obs.add_taxonomy(_process_taxonomy)
    flavio.classes.Prediction(_obs_name, br_taumull_fct(lep))
