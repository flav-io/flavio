r"""Functions for lepton flavour violating $\tau\to \ell_1\ell_2\ell_3$ decays."""

import flavio
from math import pi, log, sqrt


# names of LFV sectors in WCxf
wcxf_sector_names = {('tau', 'mu'): 'mutau',
                     ('tau', 'e'): 'taue',
                     ('mu', 'e'): 'mue',
                     ('tau', 'e', 'mu', 'e'): 'etauemu',
                     ('tau', 'mu', 'e', 'mu'): 'muemutau', }


def _BR_taumuee(mtau, me, wc):
    # (22) of hep-ph/0404211
    flavio.citations.register("Brignole:2004ah")
    return (abs(wc['CVLL'])**2 + abs(wc['CVLR'])**2 + abs(wc['CVRL'])**2 + abs(wc['CVRR'])**2
            + 1 / 4 * (abs(wc['CSLL'])**2 + abs(wc['CSLR'])**2 + abs(wc['CSRL'])**2 + abs(wc['CSRR'])**2)
            + 12 * (abs(wc['CTLL'])**2 + abs(wc['CTRR'])**2)
            + 8 * (wc['C7'] * (wc['CVLL'] + wc['CVLR']).conjugate()
                 + wc['C7p'] * (wc['CVRL'] + wc['CVRR']).conjugate()).real
            + 32 * (abs(wc['C7'])**2 + abs(wc['C7p'])**2) * (log(mtau**2 / me**2) - 3))


def _BR_tau3mu(mtau, mmu, wc):
    # (23) of hep-ph/0404211
    # (117) of hep-ph/9909265
    flavio.citations.register("Brignole:2004ah")
    flavio.citations.register("Kuno:1999jp")
    return (2 * abs(wc['CVLL'])**2 + abs(wc['CVLR'])**2 + abs(wc['CVRL'])**2 + 2 * abs(wc['CVRR'])**2
            + 1 / 8 * (abs(wc['CSLL'])**2 + abs(wc['CSRR'])**2)
            + 8 * (wc['C7'] * (2 * wc['CVLL'] + wc['CVLR']).conjugate()
                          + wc['C7p'] * (wc['CVRL'] + 2 * wc['CVRR']).conjugate()).real
            + 32 * (abs(wc['C7'])**2 + abs(wc['C7p'])**2)
            * (log(mtau**2 / mmu**2) - 11 / 4))


def _BR_taumuemu(wc):
    r"""Function for $\Delta L=2$ decays like $\tau^-\to \mu^- e^+ \mu^-$."""
    return (2 * abs(wc['CVLL'])**2 + abs(wc['CVLR'])**2 + abs(wc['CVRL'])**2 + 2 * abs(wc['CVRR'])**2
            + 1 / 8 * (abs(wc['CSLL'])**2 + abs(wc['CSRR'])**2))


def wc_eff(wc_obj, par, scale, l0, l1, l2, l3, nf_out=4):
    r"""Get the effective Wilson coefficients for the $l_0^-\to l_1^-l_2^+l_3^-$
    transition as a dictionary."""
    if l2 == l3:
        sector = wcxf_sector_names[l0, l1]
    else:
        sector = wcxf_sector_names[l0, l1, l2, l3]
    alpha = flavio.physics.running.running.get_alpha_e(par, scale, nf_out=4)
    e = sqrt(4 * pi * alpha)
    ml0 = par['m_' + l0]
    wc = wc_obj.get_wc(sector, scale, par, nf_out=nf_out)
    wceff = {}
    if (l0, l1, l2, l3) == ('tau', 'mu', 'mu', 'mu'):
        wceff['C7'] = e / ml0 * wc['Cgamma_taumu']
        wceff['C7p'] = e / ml0 * wc['Cgamma_mutau'].conjugate()
        wceff['CVLL'] = wc['CVLL_mumutaumu']
        wceff['CVLR'] = wc['CVLR_taumumumu']
        wceff['CVRL'] = wc['CVLR_mumutaumu']
        wceff['CVRR'] = wc['CVRR_mumutaumu']
        wceff['CSRR'] = wc['CSRR_mumutaumu']
        wceff['CSLL'] = wc['CSRR_mumumutau'].conjugate()
    elif (l0, l1, l2, l3) == ('tau', 'e', 'e', 'e'):
        wceff['C7'] = e / ml0 * wc['Cgamma_taue']
        wceff['C7p'] = e / ml0 * wc['Cgamma_etau'].conjugate()
        wceff['CVLL'] = wc['CVLL_eetaue']
        wceff['CVLR'] = wc['CVLR_taueee']
        wceff['CVRL'] = wc['CVLR_eetaue']
        wceff['CVRR'] = wc['CVRR_eetaue']
        wceff['CSRR'] = wc['CSRR_eetaue']
        wceff['CSLL'] = wc['CSRR_eeetau'].conjugate()
    elif (l0, l1, l2, l3) == ('tau', 'mu', 'e', 'e'):
        wceff['C7'] = e / ml0 * wc['Cgamma_taumu']
        wceff['C7p'] = e / ml0 * wc['Cgamma_mutau'].conjugate()
        wceff['CVLL'] = wc['CVLL_eetaumu']
        wceff['CVLR'] = wc['CVLR_taumuee']
        wceff['CVRL'] = wc['CVLR_eetaumu']
        wceff['CVRR'] = wc['CVRR_eetaumu']
        wceff['CSRR'] = wc['CSRR_eetaumu'] - wc['CSRR_taueemu'] / 2
        wceff['CSLL'] = wc['CSRR_eemutau'].conjugate() - wc['CSRR_mueetau'].conjugate() / 2
        wceff['CSLR'] = -2 * wc['CVLR_taueemu']
        wceff['CSRL'] = -2 * wc['CVLR_mueetau'].conjugate()
        wceff['CTLL'] = -wc['CSRR_mueetau'].conjugate() / 8
        wceff['CTRR'] = -wc['CSRR_taueemu'] / 8
    elif (l0, l1, l2, l3) == ('tau', 'e', 'mu', 'mu'):
        wceff['C7'] = e / ml0 * wc['Cgamma_taue']
        wceff['C7p'] = e / ml0 * wc['Cgamma_etau'].conjugate()
        wceff['CVLL'] = wc['CVLL_muetaumu']
        wceff['CVLR'] = wc['CVLR_tauemumu']
        wceff['CVRL'] = wc['CVLR_mumutaue']
        wceff['CVRR'] = wc['CVRR_muetaumu']
        wceff['CSRR'] = wc['CSRR_tauemumu'] - wc['CSRR_muetaumu'] / 2
        wceff['CSLL'] = wc['CSRR_mumuetau'].conjugate() - wc['CSRR_emumutau'].conjugate() / 2
        wceff['CSLR'] = -2 * wc['CVLR_taumumue']
        wceff['CSRL'] = -2 * wc['CVLR_muetaumu']
        wceff['CTLL'] = -wc['CSRR_emumutau'].conjugate() / 8
        wceff['CTRR'] = -wc['CSRR_muetaumu'] / 8
    elif (l0, l1, l2, l3) == ('mu', 'e', 'e', 'e'):
        wceff['C7'] = e / ml0 * wc['Cgamma_mue']
        wceff['C7p'] = e / ml0 * wc['Cgamma_emu'].conjugate()
        wceff['CVLL'] = wc['CVLL_eemue']
        wceff['CVLR'] = wc['CVLR_mueee']
        wceff['CVRL'] = wc['CVLR_eemue']
        wceff['CVRR'] = wc['CVRR_eemue']
        wceff['CSRR'] = wc['CSRR_eemue']
        wceff['CSLL'] = wc['CSRR_eeemu'].conjugate()
    elif (l0, l1, l2, l3) == ('tau', 'e', 'mu', 'e'):
        wceff['CVLL'] = wc['CVLL_muetaue']
        wceff['CVLR'] = wc['CVLR_tauemue']
        wceff['CVRL'] = wc['CVLR_muetaue']
        wceff['CVRR'] = wc['CVRR_muetaue']
        wceff['CSRR'] = wc['CSRR_muetaue']
        wceff['CSLL'] = wc['CSRR_emuetau'].conjugate()
    elif (l0, l1, l2, l3) == ('tau', 'mu', 'e', 'mu'):
        wceff['CVLL'] = wc['CVLL_muemutau'].conjugate()
        wceff['CVLR'] = wc['CVLR_taumuemu']
        wceff['CVRL'] = wc['CVLR_muemutau'].conjugate()
        wceff['CVRR'] = wc['CVRR_muemutau'].conjugate()
        wceff['CSRR'] = wc['CSRR_muemutau'].conjugate()
        wceff['CSLL'] = wc['CSRR_emutaumu']
    else:
        raise ValueError("Decay {}-->{}-{}+{}- not implemented".format(l0, l1, l2, l3))
    return wceff


def BR_taul1l2l3(wc_obj, par, l1, l2, l3):
    r"""Branching ratio of $\tau^-\to\ell_1^-\ell_2^+\ell_3^-$."""
    scale = flavio.config['renormalization scale']['taudecays']
    # cf. (22, 23) of hep-ph/0404211
    wceff = wc_eff(wc_obj, par, scale, 'tau', l1, l2, l3, nf_out=4)
    if (l1, l2, l3) == ('mu', 'e', 'e'):
        br_wc = _BR_taumuee(par['m_tau'], par['m_e'], wceff)
    elif (l1, l2, l3) == ('e', 'mu', 'mu'):
        br_wc = _BR_taumuee(par['m_tau'], par['m_mu'], wceff)
    elif (l1, l2, l3) == ('mu', 'mu', 'mu'):
        br_wc = _BR_tau3mu(par['m_tau'], par['m_mu'], wceff)
    elif (l1, l2, l3) == ('e', 'e', 'e'):
        br_wc = _BR_tau3mu(par['m_tau'], par['m_e'], wceff)
    elif (l1, l2, l3) == ('e', 'mu', 'e'):
        br_wc = _BR_taumuemu(wceff)
    elif (l1, l2, l3) == ('mu', 'e', 'mu'):
        br_wc = _BR_taumuemu(wceff)
    pre_br = par['tau_tau'] * par['m_tau']**5 / 192 / 8 / pi**3
    return pre_br * br_wc


# function returning function needed for prediction instance
def br_taul1l2l3_fct(l1, l2, l3):
    def f(wc_obj, par):
        return BR_taul1l2l3(wc_obj, par, l1, l2, l3)
    return f


# Observable and Prediction instances
_tex = {'e': 'e', 'mu': r'\mu'}

for (l1, l2, l3) in [('mu', 'e', 'e'), ('mu', 'mu', 'mu'),
                     ('e', 'e', 'e'), ('e', 'mu', 'mu'),
                     ('e', 'mu', 'e'), ('mu', 'e', 'mu')]:
    _process_tex = r"\tau^-\to " + _tex[l1] + r"^-" + _tex[l2] + r"^+" + _tex[l3] + r"^-"
    _process_taxonomy = r'Process :: $\tau$ lepton decays :: LFV decays :: $\tau\to \ell^\prime\ell\ell$ :: $' + _process_tex + r"$"

    _obs_name = "BR(tau->" + l1 + l2 + l3 + ")"
    _obs = flavio.classes.Observable(_obs_name)
    _obs.set_description(r"Branching ratio of $" + _process_tex + r"$")
    _obs.tex = r"$\text{BR}(" + _process_tex + r")$"
    _obs.add_taxonomy(_process_taxonomy)
    flavio.classes.Prediction(_obs_name, br_taul1l2l3_fct(l1, l2, l3))
