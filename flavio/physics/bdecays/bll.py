r"""Functions for the branching ratios and effective lifetimes of the leptonic
decays $B_q \to \ell^+\ell^-$, where $q=d$ or $s$ and $\ell=e$, $\mu$. or
$\tau$."""

from math import pi,sqrt
from flavio.physics import ckm
from flavio.physics.running import running
from flavio.physics.bdecays.common import meson_quark, lambda_K
from flavio.classes import Observable, Prediction
from flavio.config import config
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict


def br_lifetime_corr(y, ADeltaGamma):
    r"""Correction factor relating the experimentally measured branching ratio
    (time-integrated) to the theoretical one (instantaneous), see e.g. eq. (8)
    of arXiv:1204.1735.

    Parameters
    ----------

    - `y`: relative decay rate difference, $y_q = \tau_{B_q} \Delta\Gamma_q /2$
    - `ADeltaGamma`: $A_{\Delta\Gamma_q}$ as defined, e.g., in arXiv:1204.1735

    Returns
    -------

    $\frac{1-y_q^2}{1+A_{\Delta\Gamma_q} y_q}$
    """
    return (1 - y**2)/(1 + ADeltaGamma*y)

def amplitudes(par, wc, B, l1, l2):
    r"""Amplitudes P and S entering the $B_q\to\ell_1^+\ell_2^-$ observables.

    Parameters
    ----------

    - `par`: parameter dictionary
    - `B`: should be `'Bs'` or `'B0'`
    - `l1` and `l2`: should be `'e'`, `'mu'`, or `'tau'`

    Returns
    -------

    `(P, S)` where for the special case `l1 == l2` one has

    - $P = \frac{2m_\ell}{m_{B_q}} (C_{10}-C_{10}') + m_{B_q} (C_P-C_P')$
    - $S = m_{B_q} (C_S-C_S')$
    """
    # masses
    ml1 = par['m_'+l1]
    ml2 = par['m_'+l2]
    mB = par['m_'+B]
    # Wilson coefficients
    qqll = meson_quark[B] + l1 + l2
    # For LFV expressions see arXiv:1602.00881 eq. (5)
    C9m = wc['C9_'+qqll] - wc['C9p_'+qqll] # only relevant for l1 != l2!
    C10m = wc['C10_'+qqll] - wc['C10p_'+qqll]
    CPm = wc['CP_'+qqll] - wc['CPp_'+qqll]
    CSm = wc['CS_'+qqll] - wc['CSp_'+qqll]
    P = (ml2 + ml1)/mB * C10m + mB * CPm
    S = (ml2 - ml1)/mB * C9m + mB * CSm
    return P, S

def ADeltaGamma(par, wc, B, lep):
    P, S = amplitudes(par, wc, B, lep, lep)
    # cf. eq. (17) of arXiv:1204.1737
    return ((P**2).real - (S**2).real)/(abs(P)**2 + abs(S)**2)

def br_inst(par, wc, B, l1, l2):
    r"""Branching ratio of $B_q\to\ell_1^+\ell_2^-$ in the absence of mixing.

    Parameters
    ----------

    - `par`: parameter dictionary
    - `B`: should be `'Bs'` or `'B0'`
    - `lep`: should be `'e'`, `'mu'`, or `'tau'`
    """
    # paramaeters
    GF = par['GF']
    alphaem = running.get_alpha(par, 4.8)['alpha_e']
    ml1 = par['m_'+l1]
    ml2 = par['m_'+l2]
    mB = par['m_'+B]
    tauB = par['tau_'+B]
    fB = par['f_'+B]
    # appropriate CKM elements
    if B == 'Bs':
        xi_t = ckm.xi('t','bs')(par)
    elif B == 'B0':
        xi_t = ckm.xi('t','bd')(par)
    N = xi_t * 4*GF/sqrt(2) * alphaem/(4*pi)
    beta = sqrt(lambda_K(mB**2,ml1**2,ml2**2))/mB**2
    beta_p = sqrt(1 - (ml1 + ml2)**2/mB**2)
    beta_m = sqrt(1 - (ml1 - ml2)**2/mB**2)
    prefactor = abs(N)**2 / 32. / pi * mB**3 * tauB * beta * fB**2
    P, S = amplitudes(par, wc, B, l1, l2)
    return prefactor * ( beta_m**2 * abs(P)**2 + beta_p**2 * abs(S)**2 )

def br_timeint(par, wc, B, l1, l2):
    r"""Time-integrated branching ratio of $B_q\to\ell^+\ell^-$."""
    if l1 != l2:
        raise ValueError("Time-integrated branching ratio only defined for equal lepton flavours")
    lep = l1
    br0 = br_inst(par, wc, B, lep, lep)
    y = par['DeltaGamma/Gamma_'+B]/2.
    ADG = ADeltaGamma(par, wc, B, lep)
    corr = br_lifetime_corr(y, ADG)
    return br0 / corr

def bqll_obs(function, wc_obj, par, B, l1, l2):
    scale = config['renormalization scale']['bll']
    label = meson_quark[B]+l1+l2
    if l1 == l2:
        # include SM contributions for LF conserving decay
        wc = wctot_dict(wc_obj, label, scale, par)
    else:
        wc = wc_obj.get_wc(label, scale, par)
    return function(par, wc, B, l1, l2)

def bqll_obs_function(function, B, l1, l2):
    return lambda wc_obj, par: bqll_obs(function, wc_obj, par, B, l1, l2)


# Bs -> l+l- effective lifetime

def tau_ll(wc, par, B, lep):
    r"""Effective B->l+l- lifetime as defined in eq. (26) of arXiv:1204.1737 .
    This formula one either gets by integrating eq. (21) or by inverting eq. (27) of arXiv:1204.1737.

    Parameters
    ----------

    - `wc`         : dict of Wilson coefficients
    - `par`        : parameter dictionary
    - `B`          : should be `'Bs'` or `'B0'`
    - `lep`        : lepton: 'e', 'mu' or 'tau'

    Returns
    -------

    $-\frac{\tau_{B_s} \left(y_s^2+2 A_{\Delta\Gamma_q} ys+1\right)}{\left(ys^2-1\right) (A_{\Delta\Gamma_q} ys+1)}$
    """
    ADG    = ADeltaGamma(par, wc, B, lep)
    y      = .5*par['DeltaGamma/Gamma_'+B]
    tauB   = par['tau_'+B]
    return -(((1 + y**2 + 2*y*ADG)*tauB)/((-1 + y**2)*(1 + y*ADG)))

def tau_ll_func(wc_obj, par, B, lep):
    scale = config['renormalization scale']['bll']
    label = meson_quark[B]+lep+lep
    wc = wctot_dict(wc_obj, label, scale, par)
    return tau_ll(wc, par, B, lep)

# Observable and Prediction instances

_tex = {'e': 'e', 'mu': '\mu', 'tau': r'\tau'}
for l in ['e', 'mu', 'tau']:
    # For the B^0 decay, we take the time-integrated branching ratio
    _obs_name = "BR(Bs->"+l+l+")"
    _obs = Observable(_obs_name)
    _obs.set_description(r"Time-integrated branching ratio of $B_s\to "+_tex[l]+"^+"+_tex[l]+"^-$.")
    _obs.tex = r"$\overline{\text{BR}}(B_s\to "+_tex[l]+"^+"+_tex[l]+"^-)$."
    Prediction(_obs_name, bqll_obs_function(br_timeint, 'Bs', l, l))

    # For the B^0 decay, we take the prompt branching ratio since DeltaGamma is negligible
    _obs_name = "BR(Bd->"+l+l+")"
    _obs = Observable(_obs_name)
    _obs.set_description(r"Branching ratio of $B^0\to "+_tex[l]+"^+"+_tex[l]+"^-$.")
    _obs.tex = r"$\text{BR}(B^0\to "+_tex[l]+"^+"+_tex[l]+"^-)$."
    Prediction(_obs_name, bqll_obs_function(br_inst, 'B0', l, l))

    # Add the effective lifetimes for Bs
    _obs_name = 'tau_'+l+l
    _obs = Observable(_obs_name)
    _obs.set_description(r"Effective lifetime for $B_s \to "+_tex[l]+"^+"+_tex[l]+"^-$.")
    _obs.tex = r"$\tau_{B_s \to " +_tex[l] +_tex[l] + "}$."
    if l=='e':
        Prediction(_obs_name, lambda wc_obj, par: tau_ll_func(wc_obj, par, 'Bs', 'e'))
    if l=='mu':
        Prediction(_obs_name, lambda wc_obj, par: tau_ll_func(wc_obj, par, 'Bs', 'mu'))
    if l=='tau':
        Prediction(_obs_name, lambda wc_obj, par: tau_ll_func(wc_obj, par, 'Bs', 'tau'))

_tex_B = {'B0': r'\bar B^0', 'Bs': r'\bar B_s'}
_tex_lfv = {'emu': r'e^+\mu^-', 'mue': r'\mu^+e^-',
    'taue': r'\tau^+e^-', 'etau': r'e^+\tau^-',
    'taumu': r'\tau^+\mu^-', 'mutau': r'\mu^+\tau^-'}
for ll in [('e','mu'), ('mu','e'), ('e','tau'), ('tau','e'), ('mu','tau'), ('tau','mu')]:
    for B in ['Bs', 'B0']:
        _obs_name = "BR("+B+"->"+''.join(ll)+")"
        _obs = Observable(_obs_name)
        _obs.set_description(r"Branching ratio of $"+_tex_B[B]+r"\to "+_tex_lfv[''.join(ll)]+r"$")
        _obs.tex = r"$\text{BR}("+_tex_B[B]+r"\to "+_tex_lfv[''.join(ll)]+r"$)"
        Prediction(_obs_name, bqll_obs_function(br_inst, B, ll[0], ll[1]))
