r"""Functions for the anomalous magnetic moments of leptons"""

import flavio
from math import sqrt, pi


def al(wc_obj, par, l, scale):
    r"""Anomalous magnetic moment of lepton `l` at scale `scale`,
    $a_\ell = (g_\ell - 2) / 2$."""
    wc = wc_obj.get_wc(sector='dF=0', scale=scale, par=par, eft='WET-3', basis='flavio')
    # the covariant derivative contains d_mu + eta i e Q A, where Q=-1
    # the Lagrangian contains -eta * e*Q*a/(4*m) (...)
    # the effective Lagrangian contains p * ReC7 (...)
    # with p = 4*GF/sqrt(2)*e/(16*pi**2)*m
    # thus a = -eta*4*m*p/e/Q * ReC7 = pre * ReC7
    GF = par['GF']
    m = par['m_' + l]
    pre = m**2 * GF / sqrt(2) / pi**2
    C7 = wc['C7_' + 2 * l]  # e.g. C7_mumu
    aSM = par['a_{} SM'.format(l)]  # e.g. a_mu SM
    return aSM + pre * C7.real


def amu(wc_obj, par):
    r"""Anomalous magnetic moment of muon,
    $a_\mu = (g_\mu - 2) / 2$."""
    scale = flavio.config['renormalization scale']['mudecays']
    return al(wc_obj, par, l='mu', scale=scale)


def make_observable(l, ltex, lname, lfunc):
    """Instantiate the observable and add metadata."""
    obs_name = "a_" + l
    obs = flavio.classes.Observable(obs_name)
    obs.set_description(r"Anomalous magnetic moment of the " + lname)
    obs.tex = r"$a_{}$".format(ltex)
    obs.add_taxonomy(r"Process :: Dipole moments :: Lepton anomalous magnetic moments :: $a_{}$".format(ltex))
    flavio.classes.Prediction(obs_name, lfunc)


make_observable('mu', r'\mu', 'muon', amu)
