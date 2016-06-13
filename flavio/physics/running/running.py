"""Functions to solve the renormalization group equations (RGEs) numerically."""

from flavio.physics.running import betafunctions
from flavio.physics.running import masses
from scipy.integrate import odeint
import numpy as np
from functools import lru_cache
from flavio.config import config

def rg_evolve(initial_condition, derivative, scale_in, scale_out):
    sol = odeint(derivative, initial_condition, [scale_in, scale_out])
    return sol[1]

def rg_evolve_sm(initial_condition, derivative_nf, scale_in, scale_out, nf_out=None):
    if scale_in == scale_out:
        # no need to run!
        return initial_condition
    if scale_out < 0.1:
        raise ValueError('RG evolution below the strange threshold not implemented.')
    return _rg_evolve_sm(tuple(initial_condition), derivative_nf, scale_in, scale_out, nf_out)

@lru_cache(maxsize=config['settings']['cache size'])
def _rg_evolve_sm(initial_condition, derivative_nf, scale_in, scale_out, nf_out):
    # quark mass thresholds
    thresholds = {
        3: 0.1,
        4: config['RGE thresholds']['mc'],
        5: config['RGE thresholds']['mb'],
        6: config['RGE thresholds']['mt'],
        7: np.inf,
        }
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

def get_alpha(par, scale, nf_out=None):
    r"""Get the running $\overline{\mathrm{MS}}$ $\alpha_s$ and $\alpha_e$
    at the specified scale.
    """
    alpha_in = [par[('alpha_s')], par[('alpha_e')]]
    scale_in = par['m_Z']
    alpha_out = rg_evolve_sm(alpha_in, betafunctions.betafunctions_qcd_qed_nf, scale_in, scale, nf_out=nf_out)
    return dict(zip(('alpha_s','alpha_e'),alpha_out))


def _derivative_mq(x, mu, nf):
    d_alphas = betafunctions.beta_qcd_qed([x[0],0], mu, nf)[0] # only alpha_s
    d_m = masses.gamma_qcd(x[1], x[0], mu, nf)
    return [ d_alphas, d_m ]
def _derivative_mq_nf(nf):
    return lambda x, mu: _derivative_mq(x, mu, nf)


def get_mq(par, m_in, scale_in, scale_out, nf_out=None):
    alphas_in = get_alpha(par, scale_in, nf_out=nf_out)['alpha_s']
    x_in = [alphas_in, m_in]
    sol = rg_evolve_sm(x_in, _derivative_mq_nf, scale_in, scale_out, nf_out=nf_out)
    return sol[1]


def get_mb(par, scale, nf_out=None):
    r"""Get the running $b$ quark mass at the specified scale."""
    m = par['m_b']
    return get_mq(par=par, m_in=m, scale_in=m, scale_out=scale, nf_out=nf_out)

def get_mc(par, scale, nf_out=None):
    r"""Get the running $c$ quark mass at the specified scale."""
    m = par['m_c']
    return get_mq(par=par, m_in=m, scale_in=m, scale_out=scale, nf_out=nf_out)

def get_mu(par, scale, nf_out=None):
    r"""Get the running $u$ quark mass at the specified scale."""
    m = par['m_u']
    return get_mq(par=par, m_in=m, scale_in=2.0, scale_out=scale, nf_out=nf_out)

def get_md(par, scale, nf_out=None):
    r"""Get the running $d$ quark mass at the specified scale."""
    m = par['m_d']
    return get_mq(par=par, m_in=m, scale_in=2.0, scale_out=scale, nf_out=nf_out)

def get_ms(par, scale, nf_out=None):
    r"""Get the running $s$ quark mass at the specified scale."""
    m = par['m_s']
    return get_mq(par=par, m_in=m, scale_in=2.0, scale_out=scale, nf_out=nf_out)

def get_mc_pole(par, nl=2): # for mc, default to 2-loop conversion only due to renormalon ambiguity!
    r"""Get the $c$ quark pole mass, using the 2-loop conversion
    formula from the $\overline{\mathrm{MS}}$ mass."""
    mcmc = par['m_c']
    alpha_s = get_alpha(par, mcmc)['alpha_s']
    return _get_mc_pole(mcmc=mcmc, alpha_s=alpha_s, nl=nl)

# cached version
@lru_cache(maxsize=32)
def _get_mc_pole(mcmc, alpha_s, nl):
    return masses.mMS2mOS(MS=mcmc, Nf=4, asmu=alpha_s, Mu=mcmc, nl=nl)

def get_mb_pole(par, nl=2): # for mb, default to 2-loop conversion only due to renormalon ambiguity!
    r"""Get the $b$ quark pole mass, using the 2-loop conversion
    formula from the $\overline{\mathrm{MS}}$ mass."""
    mbmb = par['m_b']
    alpha_s = get_alpha(par, mbmb)['alpha_s']
    return _get_mb_pole(mbmb=mbmb, alpha_s=alpha_s, nl=nl)

# cached version
@lru_cache(maxsize=32)
def _get_mb_pole(mbmb, alpha_s, nl):
    return masses.mMS2mOS(MS=mbmb, Nf=5, asmu=alpha_s, Mu=mbmb, nl=nl)

def get_mt(par, scale):
    r"""Get the running top quark mass at the specified scale."""
    mt_pole = par['m_t']
    alpha_s = get_alpha(par, scale)['alpha_s']
    return masses.mOS2mMS(mOS=mt_pole, Nf=6, asmu=alpha_s, Mu=scale, nl=3)

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
        d_c_real = d_c.view(np.float)
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
    c_in_real = np.asarray(c_in, dtype=complex).view(np.float)
    x_in = np.append(c_in_real, [alpha_in['alpha_s'], alpha_in['alpha_e']])
    sol = rg_evolve_sm(x_in, derivative_nf, scale_in, scale_out, nf_out=nf_out)
    c_out = sol[:-2]
    return c_out.view(np.complex)
