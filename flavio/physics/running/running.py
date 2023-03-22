"""Functions to solve the renormalization group equations (RGEs) numerically."""

from flavio.physics.running import betafunctions
from flavio.physics.running import masses
from scipy.integrate import odeint
import numpy as np
from functools import lru_cache
from flavio.config import config
import copy
from math import log, pi
from wilson.util import qcd
import rundec

# quark mass thresholds
def _thresholds():
    thresholds = {
        3: 0.1,
        4: config['RGE thresholds']['mc'],
        5: config['RGE thresholds']['mb'],
        6: config['RGE thresholds']['mt'],
        7: np.inf,
        }
    return thresholds

def rg_evolve(initial_condition, derivative, scale_in, scale_out):
    sol = odeint(derivative, initial_condition, [scale_in, scale_out])
    return sol[1]

def rg_evolve_sm(initial_condition, derivative_nf, scale_in, scale_out, nf_out=None):
    if scale_in == scale_out:
        # no need to run!
        # However, return a copy to prevent accidentally changing the initial condition
        return copy.deepcopy(initial_condition)
    if scale_out < 0.1:
        raise ValueError('RG evolution below the strange threshold not implemented.')
    return _rg_evolve_sm(tuple(initial_condition), derivative_nf, scale_in, scale_out, nf_out)

@lru_cache(maxsize=config['settings']['cache size'])
def _rg_evolve_sm(initial_condition, derivative_nf, scale_in, scale_out, nf_out):
    thresholds = _thresholds()
    if scale_in > scale_out: # running DOWN
        # set initial values and scales
        initial_nf = initial_condition
        scale_in_nf = scale_in
        for nf in (6,5,4,3):
            if scale_in <= thresholds[nf]:
                continue
            if nf_out is not None and nf == nf_out:
                scale_stop = scale_out
            else:
                # run either to next threshold or to final scale, whichever is closer
                scale_stop = max(thresholds[nf], scale_out)
            sol = rg_evolve(initial_nf, derivative_nf(nf), scale_in_nf, scale_stop)
            if scale_stop == scale_out:
                return sol
            initial_nf = sol
            scale_in_nf = thresholds[nf]
    elif scale_in < scale_out: # running UP
        # set initial values and scales
        initial_nf = initial_condition
        scale_in_nf = scale_in
        for nf in (3,4,5,6):
            if nf < 6 and scale_in >= thresholds[nf+1]:
                continue
             # run either to next threshold or to final scale, whichever is closer
            scale_stop = min(thresholds[nf+1], scale_out)
            sol = rg_evolve(initial_nf, derivative_nf(nf), scale_in_nf, scale_stop)
            if scale_stop == scale_out:
                return sol
            initial_nf = sol
            scale_in_nf = thresholds[nf+1]
    return sol


@lru_cache(maxsize=config['settings']['cache size'])
def run_alpha_e(alpha_e_in, scale_in, scale_out, n_u, n_d, n_e):
    """Get the electromagnetic fine structure constant at the scale `scale_out`,
    given its value at the scale `scale_in` and running with `n_u` dynamical
    up quark flavours, `n_d` dynamical down quark flavours, and `n_e` dynamical
    charged lepton flavours."""
    if scale_out == scale_in:
        # nothing to do
        return alpha_e_in
    beta0 = -4/3 * (4 * n_u / 3 + n_d / 3 + n_e)  # -4/3 * sum Q_f^2 N_f
    return alpha_e_in / (1 + alpha_e_in * beta0 * log(scale_out/scale_in) / (2 * pi))


def get_nf(scale):
    """Guess the number of quark flavours based on the scale, if not
    specified manually."""
    mt = config['RGE thresholds']['mt']
    mb = config['RGE thresholds']['mb']
    mc = config['RGE thresholds']['mc']
    if scale >= mt:
        return 6
    elif mb <= scale < mt:
        return 5
    elif mc <= scale < mb:
        return 4
    elif scale < mc:
        return 3
    else:
        raise ValueError("Unexpected value: scale={}".format(scale))


def get_alpha_e(par, scale, nf_out=None):
    r"""Get the running $\overline{\mathrm{MS}}$ fine-structure constant
    $\alpha_e$ at the specified scale."""
    nf = nf_out or get_nf(scale)
    aeMZ = par['alpha_e']
    MZ = 91.1876  # m_Z treated as a constant here
    mt = config['RGE thresholds']['mt']
    mb = config['RGE thresholds']['mb']
    mc = config['RGE thresholds']['mc']
    if nf == 5:
        return run_alpha_e(aeMZ, MZ, scale, n_u=2, n_d=3, n_e=3)
    elif nf == 6:
        aemt = run_alpha_e(aeMZ, MZ, mt, n_u=2, n_d=3, n_e=3)
        return run_alpha_e(aemt, mt, scale, n_u=3, n_d=3, n_e=3)
    elif nf == 4:
            aemb = run_alpha_e(aeMZ, MZ, mb, n_u=2, n_d=3, n_e=3)
            return run_alpha_e(aemb, mb, scale, n_u=2, n_d=2, n_e=3)
    elif nf == 3:
        aemb = run_alpha_e(aeMZ, MZ, mb, n_u=2, n_d=3, n_e=3)
        aemc = run_alpha_e(aemb, mb, mc, n_u=2, n_d=2, n_e=3)
        return run_alpha_e(aemc, mc, scale, n_u=1, n_d=2, n_e=2)
    else:
        raise ValueError("Invalid value: nf_out={}".format(nf_out))


def get_alpha_s(par, scale, nf_out=None):
    r"""Get the running $\overline{\mathrm{MS}}$ QCD coupling constant
    $\alpha_s$ at the specified scale."""
    nf = nf_out or get_nf(scale)
    return qcd.alpha_s(scale=scale, f=nf, alphasMZ=par['alpha_s'])


def get_alpha(par, scale, nf_out=None):
    r"""Get the running $\overline{\mathrm{MS}}$ $\alpha_s$ and $\alpha_e$
    at the specified scale.
    """
    return {'alpha_e' : get_alpha_e(par, scale, nf_out=nf_out),
            'alpha_s' : get_alpha_s(par, scale, nf_out=nf_out)}


def get_mb(par, scale, nf_out=None):
    r"""Get the running $b$ quark mass at the specified scale."""
    nf = nf_out or get_nf(scale)
    return qcd.m_b(mbmb=par['m_b'], scale=scale, f=nf, alphasMZ=par['alpha_s'])


def get_mc(par, scale, nf_out=None):
    r"""Get the running $c$ quark mass at the specified scale."""
    nf = nf_out or get_nf(scale)
    return qcd.m_c(mcmc=par['m_c'], scale=scale, f=nf, alphasMZ=par['alpha_s'])


def get_mu(par, scale, nf_out=None):
    r"""Get the running $u$ quark mass at the specified scale."""
    nf = nf_out or get_nf(scale)
    return qcd.m_s(ms2=par['m_u'], scale=scale, f=nf, alphasMZ=par['alpha_s'])


def get_md(par, scale, nf_out=None):
    r"""Get the running $d$ quark mass at the specified scale."""
    nf = nf_out or get_nf(scale)
    return qcd.m_s(ms2=par['m_d'], scale=scale, f=nf, alphasMZ=par['alpha_s'])


def get_ms(par, scale, nf_out=None):
    r"""Get the running $s$ quark mass at the specified scale."""
    nf = nf_out or get_nf(scale)
    return qcd.m_s(ms2=par['m_s'], scale=scale, f=nf, alphasMZ=par['alpha_s'])


def get_mq(q, par, scale, nf_out=None):
    fdict = {'u': get_mu,
             'c': get_mc,
             'd': get_md,
             's': get_ms,
             'b': get_mb}
    return fdict[q](par, scale, nf_out=nf_out)


def get_mc_pole(par, nl=2): # for mc, default to 2-loop conversion only due to renormalon ambiguity!
    r"""Get the $c$ quark pole mass, using the 2-loop conversion
    formula from the $\overline{\mathrm{MS}}$ mass."""
    mcmc = par['m_c']
    alpha_s = get_alpha(par, mcmc)['alpha_s']
    return _get_mc_pole(mcmc=mcmc, alpha_s=alpha_s, nl=nl)


# cached version
@lru_cache(maxsize=config['settings']['cache size'])
def _get_mc_pole(mcmc, alpha_s, nl):
    crd = rundec.CRunDec()
    return crd.mMS2mOS(mcmc, None, alpha_s, mcmc, 4, nl)


def get_mc_KS(par, scale):
    r"""Get the $c$ quark mass in the kinetic scheme."""
    mcmc = par['m_c']
    alpha_s = get_alpha(par, mcmc)['alpha_s']
    return _get_mc_KS(mcmc=mcmc, alpha_s=alpha_s, scale=scale, nl=3)


# cached version
@lru_cache(maxsize=config['settings']['cache size'])
def _get_mc_KS(mcmc, alpha_s, scale, nl):
    return masses.mMS2mKS(MS=mcmc, Nf=3, asM=alpha_s, Mu=scale, nl=nl)


def get_mb_pole(par, nl=2):  # for mb, default to 2-loop conversion only due to renormalon ambiguity!
    r"""Get the $b$ quark pole mass, using the 2-loop conversion
    formula from the $\overline{\mathrm{MS}}$ mass."""
    mbmb = par['m_b']
    alpha_s = get_alpha(par, mbmb)['alpha_s']
    return _get_mb_pole(mbmb=mbmb, alpha_s=alpha_s, nl=nl)


# cached version
@lru_cache(maxsize=config['settings']['cache size'])
def _get_mb_pole(mbmb, alpha_s, nl):
    crd = rundec.CRunDec()
    return crd.mMS2mOS(mbmb, None, alpha_s, mbmb, 5, nl)


def get_mb_KS(par, scale):
    r"""Get the $b$ quark mass in the kinetic scheme."""
    mbmb = par['m_b']
    alpha_s = get_alpha(par, mbmb)['alpha_s']
    return _get_mb_KS(mbmb=mbmb, alpha_s=alpha_s, scale=scale, nl=3)


# cached version
@lru_cache(maxsize=config['settings']['cache size'])
def _get_mb_KS(mbmb, alpha_s, scale, nl):
    # see 1107.3100 for why Nf=4
    return masses.mMS2mKS(MS=mbmb, Nf=4, asM=alpha_s, Mu=scale, nl=nl)


def get_mb_1S(par, nl=3):
    r"""Get the $b$ quark mass in the 1S scheme."""
    mbmb = par['m_b']
    alpha_s = get_alpha(par, mbmb)['alpha_s']
    return _get_mb_1S(mbmb=mbmb, alpha_s=alpha_s, scale=mbmb, nl=nl)


# cached version
@lru_cache(maxsize=config['settings']['cache size'])
def _get_mb_1S(mbmb, alpha_s, scale, nl):
    crd = rundec.CRunDec()
    return crd.mMS2m1S(mbmb, None, alpha_s, scale, 5, nl)


def get_mt(par, scale):
    r"""Get the running top quark mass at the specified scale."""
    return _get_mt(mt_pole=par['m_t'],
                   alpha_s=get_alpha_s(par, scale),
                   scale=scale)


# cached version
@lru_cache(maxsize=config['settings']['cache size'])
def _get_mt(mt_pole, alpha_s, scale):
    r"""Get the running top quark mass at the specified scale."""
    crd = rundec.CRunDec()
    return crd.mOS2mMS(mt_pole, None, alpha_s, scale, 6, 3)


def get_mt_mt(par):
    r"""Get the scale invariant top quark mass mt(mt)."""
    mt_pole = par['m_t']
    return _get_mt_mt(mt_pole=mt_pole,
                      alpha_s=get_alpha_s(par, mt_pole))


# cached version
@lru_cache(maxsize=config['settings']['cache size'])
def _get_mt_mt(mt_pole, alpha_s):
    r"""Get the scale invariant top quark mass mt(mt)."""
    crd = rundec.CRunDec()
    return crd.mOS2mSI(mt_pole, None, alpha_s, 6, 3)


def make_wilson_rge_derivative(adm):
    if adm is None:
        return None
    def derivative(x, mu, nf):
        alpha_s = x[-2]
        alpha_e = x[-1]
        c_real = x[:-2]
        c = c_real.view(np.complex)
        d_alpha = betafunctions.beta_qcd_qed([alpha_s, alpha_e], mu, nf)
        d_c = np.dot(adm(nf, alpha_s, alpha_e).T, c)/mu
        d_c_real = d_c.view(float)
        return np.append(d_c_real, d_alpha)
    def derivative_nf(nf):
        return lambda x, mu: derivative(x, mu, nf)
    return derivative_nf

def get_wilson(par, c_in, derivative_nf, scale_in, scale_out, nf_out=None):
    r"""RG evolution of a vector of Wilson coefficients.

    In terms of the anomalous dimension matrix $\gamma$, the RGE reads
    $$\mu\frac{d}{d\mu} \vec C = \gamma^T(n_f, \alpha_s, \alpha_e) \vec C$$
    """
    alpha_in = get_alpha(par, scale_in, nf_out=nf_out)
    # x is (c_1, ..., c_N, alpha_s, alpha_e)
    c_in_real = np.asarray(c_in, dtype=complex).view(float)
    x_in = np.append(c_in_real, [alpha_in['alpha_s'], alpha_in['alpha_e']])
    sol = rg_evolve_sm(x_in, derivative_nf, scale_in, scale_out, nf_out=nf_out)
    c_out = sol[:-2]
    return c_out.view(np.complex)

def get_f_perp(par, meson, scale, nf_out=None):
    r"""Get the transverse meson decay constant at a given scale.
    The argument `meson` should be one of `rho0`, `rho+`, `K*0`, `K*+`,
    `omega`, `phi`. The input value for the transverse decay constant is taken
    from `par` and is assumed to be at a scale of 2 GeV in the 3-flavour
    scheme.
    """
    thresholds = _thresholds()
    f_perp = par['f_perp_' +  meson]
    scale_start = 2
    nf = 3
    if scale == scale_start:
        return f_perp
    for nf in (3, 4, 5, 6):
        # run to final scale directly if nf equals nf_out
        if nf_out is not None and nf_out == nf:
            scale_stop = scale
        # run either to next threshold or to final scale, whichever is closer
        else:
            scale_stop = min(thresholds[nf + 1], scale)
        f_perp = _rg_factor_f_perp(par, scale_start, scale_stop, nf) * f_perp
        if scale_stop == scale:
            return f_perp
        scale_start = thresholds[nf + 1]

def _rg_factor_f_perp(par, scale_start, scale_stop, nf):
    r"""This function returns the leading order renormalization group (RG)
    factor that relates the value of the transverse meson decay constant
    at the scale `scale_stop` to its value at the scale `scale_start`.
    The number of fermion flavours to be taken into account is specified by
    `nf`. The exponent used in the RG factor can be written es `4/(3*beta_0)`,
    where `beta_0` is given by `11-2*nf/3` (cf. e.g.
    https://arxiv.org/abs/hep-lat/0301020 eq. (14)).
    """
    alpha_start = get_alpha_s(par, scale_start, nf_out=nf)
    alpha_stop = get_alpha_s(par, scale_stop, nf_out=nf)
    return (alpha_stop / alpha_start)**(4 / (33 - 2 * nf))
