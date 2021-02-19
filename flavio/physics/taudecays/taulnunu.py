r"""Functions for $\tau\to \ell \nu\nu$ decays."""

import flavio
from math import log, sqrt, pi


def F(x):
    return 1 - 8*x + 8*x**3 - x**4 - 12*x**2*log(x)

def G(x):
    return 1 + 9*x - 9*x**2 - x**3 + 6*x*log(x) + 6*x**2*log(x)

def _BR(x, CL, CR):
    return F(x) * (abs(CL)**2 + abs(CR)**2) - 4 * G(x) * (CL * CR.conjugate()).real


def GFeff(wc_obj, par):
    r"""Effective Fermi constant in the presence of new physics."""
    scale = flavio.config['renormalization scale']['mudecays']
    wc = wc_obj.get_wc('nunumue', scale, par, eft='WET-3')
    CL = wc['CVLL_numunueemu']
    CR = wc['CVLR_numunueemu']
    me = par['m_e']
    mmu = par['m_mu']
    GF = par['GF']
    x = me**2 / mmu**2
    CLSM = -4 *  GF / sqrt(2)
    r = _BR(x, CL + CLSM, CR) / _BR(x, CLSM, 0)
    return GF / sqrt(r)


def BR_taulnunu(wc_obj, par, lep, nu1, nu2):
    r"""BR of $\tau\to l nu_1\bar nu_2$ for specific neutrino flavours"""
    if lep == 'e':
        sec = 'nunutaue'
    elif lep == 'mu':
        sec = 'nunumutau'
    scale = flavio.config['renormalization scale']['taudecays']
    wc = wc_obj.get_wc(sec, scale, par, eft='WET-4')
    ml = par['m_' + lep]
    mtau = par['m_tau']
    x = ml**2 / mtau**2
    nnll = 'nu{}nu{}tau{}'.format(nu2, nu1, lep)
    try:
        CL = wc['CVLL_' + nnll]
        CR = wc['CVLR_' + nnll]
    except KeyError:
        nnll = 'nu{}nu{}{}tau'.format(nu1, nu2, lep)
        CL = wc['CVLL_' + nnll].conjugate()
        CR = wc['CVLR_' + nnll].conjugate()
    if nu1 == 'tau' and nu2 == lep:
        # SM contribution, taking into account NP in mu->enunu!
        GF = GFeff(wc_obj, par)
        CL += -4 * GF / sqrt(2)
    pre = par['tau_tau'] / 3 / 2**9 / pi**3 * mtau**5
    alpha_e = flavio.physics.running.running.get_alpha_e(par, scale, nf_out=4)
    # eq. (3) of arXiv:1310.7922
    flavio.citations.register("Pich:2013lsa")
    emcorr = 1 + alpha_e / (2 * pi) * (25 / 4 - pi**2)
    return pre * _BR(x, CL, CR) * emcorr


def BR_taulnunu_summed(wc_obj, par, lep):
    """BR of tau->lnunu summed over neutrino flavours"""
    _l =  ['e', 'mu', 'tau']
    return sum([BR_taulnunu(wc_obj, par, lep, nu1, nu2) for nu1 in _l for nu2 in _l])


# function returning function needed for prediction instance
def br_taulnunu(lep):
    def f(wc_obj, par):
        return BR_taulnunu_summed(wc_obj, par, lep)
    return f


# Observable and Prediction instances
_tex = {'e': 'e', 'mu': r'\mu'}

for lep in _tex:
    _process_tex = r"\tau^-\to " + _tex[lep] + r"^- \nu\bar\nu"
    _process_taxonomy = r'Process :: $\tau$ lepton decays :: Leptonic tree-level decays :: $\tau\to \ell\nu\bar\nu$ :: $' + _process_tex + r"$"

    _obs_name = "BR(tau->" + lep + "nunu)"
    _obs = flavio.classes.Observable(_obs_name)
    _obs.set_description(r"Branching ratio of $" + _process_tex + r"$")
    _obs.tex = r"$\text{BR}(" + _process_tex + r")$"
    _obs.add_taxonomy(_process_taxonomy)
    flavio.classes.Prediction(_obs_name, br_taulnunu(lep))
