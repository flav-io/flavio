from math import sqrt,pi
import numpy as np
from flavio.physics.bdecays.common import lambda_K, beta_l, meson_quark, meson_ff
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict
from flavio.physics.bdecays import matrixelements
from flavio.physics import ckm
from flavio.physics.bdecays.formfactors import FormFactorParametrization as FF
from flavio.config import config
from flavio.physics.running import running
from flavio.physics.bdecays.bvll import qcdf
from flavio.physics.common import conjugate_par, conjugate_wc, add_dict

"""Functions for exclusive $B\to V\ell^+\ell^-$ decays."""


def prefactor(q2, par, B, V, lep):
    GF = par['Gmu']
    scale = config['bdecays']['scale_bvll']
    alphaem = running.get_alpha(par, scale)['alpha_e']
    ml = par[('mass',lep)]
    mB = par[('mass',B)]
    mV = par[('mass',V)]
    tauB = par[('lifetime',B)]
    X = sqrt(lambda_K(mB**2,q2,mV**2))/2.
    di_dj = meson_quark[(B,V)]
    xi_t = ckm.xi('t',di_dj)(par)
    if q2 <= 4*ml**2:
        return 0
    return ( sqrt((GF**2 * alphaem**2)/(3 * 2**10 * pi**5 * mB**3)
            * q2 * 2 * X *beta_l(ml, q2)) * xi_t )


def transversity_amps_ff(q2, wc, par, B, V, lep):
    """B->Vll transversity amplitudes containing contributions proportional
    to B->V form factors.
    """
    # setting the parameters and prefactor
    ml = par[('mass',lep)]
    mB = par[('mass',B)]
    scale = config['bdecays']['scale_bvll']
    mb = running.get_mb(par, scale)
    mV = par[('mass',V)]
    X = sqrt(lambda_K(mB**2,q2,mV**2))/2.
    N = prefactor(q2, par, B, V, lep)
    # functions for the "effective" Wilson coefficients including the matrix
    # elements of 4-quark operators
    #   a) LO Q1-6
    Yq2 = matrixelements.Y(q2, wc, par, scale)
    #   b) NNLO Q1,2
    delta_C7 = matrixelements.delta_C7(par=par, wc=wc, q2=q2, scale=scale, qiqj=meson_quark[(B,V)])
    delta_C9 = matrixelements.delta_C9(par=par, wc=wc, q2=q2, scale=scale, qiqj=meson_quark[(B,V)])
    # convenient combinations of Wilson coefficients
    c7pl = wc['C7eff'] + wc['C7effp'] + delta_C7
    c7mi = wc['C7eff'] - wc['C7effp'] + delta_C7
    c9pl = wc['C9'] + wc['C9p']       + delta_C9 + Yq2
    c9mi = wc['C9'] - wc['C9p']       + delta_C9 + Yq2
    c10pl = wc['C10'] + wc['C10p']
    c10mi = wc['C10'] - wc['C10p']
    csmi = wc['CS'] - wc['CSp']
    cpmi = wc['CP'] - wc['CPp']
    # form factors
    ff = FF.parametrizations['bsz3'].get_ff(meson_ff[(B,V)], q2, par)
    # transverity amplitudes
    ta = {}
    ta['perp_L'] = N * sqrt(2)*2*X * ((c9pl - c10pl) * (ff['V']/(mB + mV)) + 2*mb/q2 * c7pl * ff['T1'])
    ta['perp_R'] = N * sqrt(2)*2*X * ((c9pl + c10pl) * (ff['V']/(mB + mV)) + 2*mb/q2 * c7pl * ff['T1'])
    ta['para_L'] = -N * sqrt(2) * (mB**2 - mV**2) * ((c9mi - c10mi) * (ff['A1']/(mB - mV)) + 2*mb/q2 * c7mi * ff['T2'])
    ta['para_R'] = -N * sqrt(2) * (mB**2 - mV**2) * ((c9mi + c10mi) * (ff['A1']/(mB - mV)) + 2*mb/q2 * c7mi * ff['T2'])
    ta['0_L'] = -N * (1/(2 * mV * sqrt(q2))) * ((c9mi - c10mi) * 16*mB*mV**2 * ff['A12']
                + 2 * mb * c7mi * 8*mB*mV**2/(mB+mV) * ff['T23'])
    ta['0_R'] = -N * (1/(2 * mV * sqrt(q2))) * ((c9mi + c10mi) * 16*mB*mV**2 * ff['A12']
                + 2 * mb * c7mi * 8*mB*mV**2/(mB+mV) * ff['T23'])
    ta['t'] = N * 1/sqrt(q2) * 2 * X * (2*c10mi + q2/(2*ml)*cpmi) * ff['A0']
    ta['S'] = -N * 2*X * csmi * ff['A0']
    return ta

def transversity_amps_qcdf(q2, wc, par, B, V, lep):
    """QCD factorization corrections to B->Vll transversity amplitudes."""
    mB = par[('mass',B)]
    mV = par[('mass',V)]
    scale = config['bdecays']['scale_bvll']
    # using the b quark pole mass here!
    mb = running.get_mb_pole(par)
    N = prefactor(q2, par, B, V, lep)
    T_perp = qcdf.T_perp(q2, par, wc, B, V, scale)
    T_para = qcdf.T_para(q2, par, wc, B, V, scale)
    ta = {}
    ta['perp_L'] = N * sqrt(2)*2 * (mB**2-q2) * mb / q2 * T_perp
    ta['perp_R'] =  ta['perp_L']
    ta['para_L'] = -ta['perp_L']
    ta['para_R'] =  ta['para_L']
    ta['0_L'] = ( N * mb * (mB**2 - q2)**2 )/(mB**2 * mV * sqrt(q2)) * T_para
    ta['0_R'] = ta['0_L']
    ta['t'] = 0
    ta['S'] = 0
    return ta

def transversity_amps(*args, **kwargs):
    return add_dict((
        transversity_amps_ff(*args, **kwargs),
        transversity_amps_qcdf(*args, **kwargs),
        ))


def transversity_amps_bar(q2, wc, par, B, V, lep):
    par_c = conjugate_par(par)
    wc_c = conjugate_wc(wc)
    return transversity_amps(q2, wc_c, par_c, B, V, lep)
