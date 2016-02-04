from flavio.physics.bdecays.common import lambda_K, beta_l
from math import sqrt, pi
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict, get_wceff
from flavio.physics.running import running
from flavio.physics.bdecays.formfactors import FormFactorParametrization as FF
from flavio.config import config
from flavio.physics.bdecays.common import lambda_K, beta_l, meson_quark, meson_ff
from flavio.physics.common import conjugate_par, conjugate_wc, add_dict
from flavio.physics.bdecays import matrixelements, angular
from flavio.physics import ckm
from flavio.physics.bdecays.bvll import qcdf

def prefactor(q2, par, B, V, lep):
    GF = par['Gmu']
    ml = par['m_'+lep]
    scale = config['renormalization scale']['bvll']
    alphaem = running.get_alpha(par, scale)['alpha_e']
    di_dj = meson_quark[(B,V)]
    xi_t = ckm.xi('t',di_dj)(par)
    if q2 <= 4*ml**2:
        return 0
    return 4*GF/sqrt(2)*xi_t*alphaem/(4*pi)

def get_ff(q2, par, B, V):
    return FF.parametrizations['bsz3'].get_ff(meson_ff[(B,V)], q2, par)

def transversity_to_helicity(ta):
    H={}
    H['0' ,'V'] = -1j * (ta['0_R'] + ta['0_L'])
    H['0' ,'A'] = -1j * (ta['0_R'] - ta['0_L'])
    H['pl' ,'V'] = 1j * ((ta['para_R'] + ta['para_L']) + (ta['perp_R'] + ta['perp_L']))/sqrt(2)
    H['pl' ,'A'] = 1j * ((ta['para_R'] - ta['para_L']) + (ta['perp_R'] - ta['perp_L']))/sqrt(2)
    H['mi' ,'V'] = 1j * ((ta['para_R'] + ta['para_L']) - (ta['perp_R'] + ta['perp_L']))/sqrt(2)
    H['mi' ,'A'] = 1j * ((ta['para_R'] - ta['para_L']) - (ta['perp_R'] - ta['perp_L']))/sqrt(2)
    return H


def transversity_amps_qcdf(q2, wc, par, B, V, lep):
    """QCD factorization corrections to B->Vll transversity amplitudes."""
    mB = par['m_'+B]
    mV = par['m_'+V]
    scale = config['renormalization scale']['bvll']
    # using the b quark pole mass here!
    mb = running.get_mb_pole(par)
    N = prefactor(q2, par, B, V, lep)/4
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

def helicity_amps_qcdf(q2, wc, par, B, V, lep):
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mV = par['m_'+V]
    X = sqrt(lambda_K(mB**2,q2,mV**2))/2.
    ta = transversity_amps_qcdf(q2, wc, par, B, V, lep)
    h = transversity_to_helicity(ta)
    return h

def helicity_amps_ff(q2, wc, par, B, V, lep):
    scale = config['renormalization scale']['bvll']
    wc_eff = get_wceff(q2, wc, par, B, V, lep, scale)
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mV = par['m_'+V]
    mb = running.get_mb(par, scale)
    N = prefactor(q2, par, B, V, lep)
    ff = get_ff(q2, par, B, V)
    h = angular.helicity_amps_v(q2, mB, mV, mb, 0, ml, ml, ff, wc_eff, N)
    return h

def helicity_amps(*args, **kwargs):
    return add_dict((
        helicity_amps_ff(*args, **kwargs),
        helicity_amps_qcdf(*args, **kwargs),
        ))

def helicity_amps_bar(q2, wc, par, B, V, lep):
    par_c = conjugate_par(par)
    wc_c = conjugate_wc(wc)
    return helicity_amps(q2, wc_c, par_c, B, V, lep)
