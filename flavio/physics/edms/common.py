"""Common functions for EDMs."""

import flavio
from math import sqrt, pi

# convert from EFT to 'nf_out'
eft_nf_out = {'WET': 5, 'WET-4': 4, 'WET-3': 3, 'WET-2': 2}


def get_m(par, f, scale, eft):
    if f in ['e', 'mu', 'tau']:
        return par['m_' + f]
    nf_out = eft_nf_out[eft]
    if f == 'u':
        return flavio.physics.running.running.get_md(par, scale, nf_out=nf_out)
    elif f == 'd':
        return flavio.physics.running.running.get_mu(par, scale, nf_out=nf_out)
    elif f == 's':
        return flavio.physics.running.running.get_ms(par, scale, nf_out=nf_out)
    else:
        raise ValueError("get_m not defined for fermion {}".format(f))


def edm_f(par, wc, f, scale, eft):
    r"""EDM of the fermion `f` at the scale `scale`.

    In terms of the Wilson coefficient of the dipole operator in the
    WCxf flavio basis, it is given as

    $$d_f = 2p \text{Im}C_7^{ff}$$

    where $p = \frac{4 G_F}\sqrt{2} \frac{e}{16\pi^2} m_f$ is the prefactor
    of the operator in the effective Lagrangian."""
    m = get_m(par, f, scale, eft)
    nf_out = eft_nf_out[eft]
    alpha = flavio.physics.running.running.get_alpha(par, scale, nf_out=nf_out)
    e = sqrt(4 * pi * alpha['alpha_e'])
    pre = 4 * par['GF'] / sqrt(2) * e / (16 * pi**2) * m
    return 2 * pre * wc['C7_{}'.format(2*f)].imag


def cedm_f(par, wc, f, scale, eft):
    r"""EDM of the quark `f` at the scale `scale`.

    In terms of the Wilson coefficient of the dipole operator in the
    WCxf flavio basis, it is given as

    $$d_f = 2p/g_s \text{Im}C_8^{ff}$$

    where $p = \frac{4 G_F}\sqrt{2} \frac{g_s}{16\pi^2} m_f$ is the prefactor
    of the operator in the effective Lagrangian."""
    m = get_m(par, f, scale, eft)
    pre_gs = 4 * par['GF'] / sqrt(2) / (16 * pi**2) * m  # p / gs
    return 2 * pre_gs * wc['C8_{}'.format(2*f)].imag
