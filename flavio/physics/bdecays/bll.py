from math import pi,sqrt
from flavio.physics.bdecays.common import wcsm
from flavio.physics import ckm

r"""Functions for the branching ratios and effective lifetimes of the leptonic
decays $B_q \to \ell^+\ell^-$, where $q=d$ or $s$ and $\ell=e$, $\mu$. or
$\tau$."""

def br_lifetime_corr(y, ADeltaGamma):
    r"""Correction factor relating the experimentally measured branching ratio
    (time-integrated) to the theoretical one (instantaneous), see e.g. eq. (8)
    of arXiv:1204.1735.

    Parameters
    ----------
    - y: relative decay rate difference, $y_q = \tau_{B_q} \Delta\Gamma_q /2$
    - ADeltaGamma: $A_{\Delta\Gamma_q}$ as defined, e.g., in arXiv:1204.1735

    Returns
    -------
    $\frac{1-y_q^2}{1+A_{\Delta\Gamma_q} y_q}$
    """
    return (1 - y**2)/(1 + ADeltaGamma*y)

def amplitudes(par, wc, B, lep):
    r"""Amplitudes P and S entering the $B_q\to\ell^+\ell^-$ observables.

    Parameters
    ----------
    - par: parameter dictionary
    - B: should be 'Bs' or 'Bd'
    - lep: should be 'e', 'mu', or 'tau'

    Returns
    -------
    P, S where
    - $P = \frac{2m_\ell}{m_{B_q}} (C_{10}-C_{10}') + m_{B_q} (C_P-C_P')$
    - $S = m_{B_q} (C_S-C_S')$
    """
    # masses
    ml = par[('mass',lep)]
    mB = par[('mass',B)]
    # Wilson coefficients
    C10m = wc['C10'] + wcsm['C10'] - wc['C10p']
    CPm = wc['CP'] - wc['CPp']
    CSm = wc['CS'] - wc['CSp']
    P = 2*ml/mB * C10m +  mB * CPm
    S = mB * CSm
    return P, S

def ADeltaGamma(par, wc, B, lep):
    P, S = amplitudes(par, wc, B, lep)
    # cf. eq. (17) of arXiv:1204.1737
    return ((P**2).real - (S**2).real)/(abs(P)**2 + abs(S)**2)

def br_inst(par, wc, B, lep):
    r"""Branching ratio of $B_q\to\ell^+\ell^-$ at $t=0$ (before mixing).

    Parameters
    ----------
    - par: parameter dictionary
    - B: should be 'Bs' or 'Bd'
    - lep: should be 'e', 'mu', or 'tau'
    """
    # paramaeters
    GF = par['Gmu']
    alphaem = par['alphaem']
    ml = par[('mass',lep)]
    mB = par[('mass',B)]
    tauB = par[('lifetime',B)]
    fB = par[('f',B)]
    # appropriate CKM elements
    if B == 'Bs':
        xi_t = ckm.xi('t','bs')(par)
    elif B == 'Bd':
        xi_t = ckm.xi('t','bd')(par)
    N = xi_t * 4*GF/sqrt(2) * alphaem/(4*pi)
    beta = sqrt(1-4*ml**2/mB**2)
    prefactor = abs(N)**2 / 32. / pi * mB**3 * tauB * beta * fB**2
    P, S = amplitudes(par, wc, B, lep)
    return prefactor * ( abs(P)**2 + beta**2 * abs(S)**2 )

def br_timeint(par, wc, B, lep):
    r"""Time-integrated branching ratio of $B_q\to\ell^+\ell^-$."""
    br0 = br_inst(par, wc, B, lep)
    y = par[('DeltaGamma/Gamma',B)]/2.
    ADG = ADeltaGamma(par, wc, B, lep)
    corr = br_lifetime_corr(y, ADG)
    return br0 / corr
