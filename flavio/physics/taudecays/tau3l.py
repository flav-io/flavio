r"""Functions for lepton flavour violating $\tau\to \ell_1\ell_2\ell_2$ decays."""

import flavio
from math import pi, log, sqrt


# names of LFV sectors in WCxf
wcxf_sector_names = {('tau', 'mu'): 'mutau',
                     ('tau', 'e'): 'taue',
                     ('mu', 'e'): 'mue', }


def _BR_taumuee(mtau, me, wc):
    # (22) of hep-ph/0404211
    return (abs(wc['CVLL'])**2 + abs(wc['CVLR'])**2 + abs(wc['CVRL'])**2 + abs(wc['CVRR'])**2
            + 1 / 4 * (abs(wc['CSLL'])**2 + abs(wc['CSLR'])**2 + abs(wc['CSRL'])**2 + abs(wc['CSRR'])**2)
            + 12 * (abs(wc['CTLL'])**2 + abs(wc['CTRR'])**2)
            + 4 * (wc['C7'] * (wc['CVLL'] + wc['CVLR']).conjugate()
                 + wc['C7p'] * (wc['CVRL'] + wc['CVRR']).conjugate()).real
            + 8 * (abs(wc['C7'])**2 + abs(wc['C7p'])**2) * (log(mtau**2 / me**2) - 3))


def _BR_tau3mu(mtau, mmu, wc):
    # (23) of hep-ph/0404211
    # (117) of hep-ph/9909265
    return (2 * abs(wc['CVLL'])**2 + abs(wc['CVLR'])**2 + abs(wc['CVRL'])**2 + 2 * abs(wc['CVRR'])**2
            + 1 / 8 * (abs(wc['CSLL'])**2 + abs(wc['CSRR'])**2)
            + 4 * (wc['C7'] * (2 * wc['CVLL'] + wc['CVLR']).conjugate()
                          + wc['C7p'] * (wc['CVRL'] + 2 * wc['CVRR']).conjugate()).real
            + 8 * (abs(wc['C7'])**2 + abs(wc['C7p'])**2)
            * (log(mtau**2 / mmu**2) - 11 / 4))


def wc_eff(wc_obj, par, scale, l0, l1, l2, l3, nf_out=4):
    r"""Get the effective Wilson coefficients for the $l_0^-\to l_1^-l_2^+l_3^-$
    transition as a dictionary."""
    if l2 == l3:
        sector = wcxf_sector_names[l0, l1]
    else:
        raise ValueError("Not implemented")
    alpha = flavio.physics.running.running.get_alpha_e(par, scale, nf_out=4)
    e = sqrt(4 * pi * alpha)
    ml0 = par['m_' + l0]
    wc = wc_obj.get_wc(sector, scale, par, nf_out=nf_out)
    wceff = {}
    if (l0, l1, l2, l3) == ('tau', 'mu', 'mu', 'mu'):
        wceff['C7'] = e / ml0 * wc['Cgamma_taumu']
        wceff['C7p'] = e / ml0 * wc['Cgamma_mutau'].conjugate()
        wceff['CVLL'] = wc['CVLL_mumutaumu']
        wceff['CVLR'] = wc['CVLR_mumutaumu']
        wceff['CVRL'] = wc['CVLR_taumumumu']
        wceff['CVRR'] = wc['CVRR_mumutaumu']
        wceff['CSRR'] = wc['CSRR_mumutaumu']
        wceff['CSLL'] = wc['CSRR_mumumutau'].conjugate()
    elif (l0, l1, l2, l3) == ('tau', 'mu', 'e', 'e'):
        wceff['C7'] = e / ml0 * wc['Cgamma_taumu']
        wceff['C7p'] = e / ml0 * wc['Cgamma_mutau'].conjugate()
        wceff['CVLL'] = wc['CVLL_eetaumu']
        wceff['CVLR'] = wc['CVLR_eetaumu']
        wceff['CVRL'] = wc['CVLR_taumuee']
        wceff['CVRR'] = wc['CVRR_eetaumu']
        wceff['CSRR'] = wc['CSRR_eetaumu'] - wc['CSRR_taueemu'] / 2
        wceff['CSLL'] = wc['CSRR_eemutau'].conjugate() - wc['CSRR_mueetau'].conjugate() / 2
        wceff['CSLR'] = -2 * wc['CVLR_taueemu']
        wceff['CSRL'] = -2 * wc['CVLR_mueetau'].conjugate()
        wceff['CTLL'] = -wc['CSRR_mueetau'].conjugate() / 8
        wceff['CTRR'] = -wc['CSRR_taueemu'] / 8
    elif (l0, l1, l2, l3) == ('mu', 'e', 'e', 'e'):
        wceff['C7'] = e / ml0 * wc['Cgamma_mue']
        wceff['C7p'] = e / ml0 * wc['Cgamma_emu'].conjugate()
        wceff['CVLL'] = wc['CVLL_eemue']
        wceff['CVLR'] = wc['CVLR_eemue']
        wceff['CVRL'] = wc['CVLR_mueee']
        wceff['CVRR'] = wc['CVRR_eemue']
        wceff['CSRR'] = wc['CSRR_eemue']
        wceff['CSLL'] = wc['CSRR_eeemu'].conjugate()
    else:
        raise ValueError("Decay {}-->{}-{}+{}- not implemented".format(l0, l1, l2, l3))
    return wceff


def BR_taumull(wc_obj, par, lep):
    r"""Branching ratio of $\tau^-\to\mu^-\ell^+\ell^-$."""
    scale = flavio.config['renormalization scale']['taudecays']
    # cf. (22, 23) of hep-ph/0404211
    wceff = wc_eff(wc_obj, par, scale, 'tau', 'mu', lep, lep, nf_out=4)
    mtau = par['m_tau']
    if lep == 'e':
        me = par['m_e']
        br_wc = _BR_taumuee(mtau, me, wceff)
    elif lep == 'mu':
        mmu = par['m_mu']
        br_wc = _BR_tau3mu(mtau, mmu, wceff)
    pre_br = par['BR(tau->mununu)'] / (8 * par['GF']**2)
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
