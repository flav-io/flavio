"""Functions for constructing the helicity amplitudes"""

from flavio.physics.bdecays.common import lambda_K, beta_l
from math import sqrt, pi
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict, get_wceff
from flavio.physics.running import running
from flavio.config import config
from flavio.physics.bdecays.common import lambda_K, beta_l, meson_quark, meson_ff
from flavio.physics.common import conjugate_par, conjugate_wc, add_dict
from flavio.physics.bdecays import matrixelements, angular
from flavio.physics import ckm
from flavio.physics.bdecays.bvll import qcdf
from flavio.classes import AuxiliaryQuantity
import warnings


def prefactor(q2, par, B, V):
    GF = par['GF']
    scale = config['renormalization scale']['bvll']
    alphaem = running.get_alpha(par, scale)['alpha_e']
    di_dj = meson_quark[(B,V)]
    xi_t = ckm.xi('t',di_dj)(par)
    return 4*GF/sqrt(2)*xi_t*alphaem/(4*pi)


def get_ff(q2, par, B, V):
    ff_name = meson_ff[(B,V)] + ' form factor'
    return AuxiliaryQuantity[ff_name].prediction(par_dict=par, wc_obj=None, q2=q2)


def helicity_amps_ff(q2, ff, wc_obj, par_dict, B, V, lep, cp_conjugate):
    par = par_dict.copy()
    if cp_conjugate:
        par = conjugate_par(par)
    scale = config['renormalization scale']['bvll']
    label = meson_quark[(B,V)] + lep + lep # e.g. bsmumu, bdtautau
    wc = wctot_dict(wc_obj, label, scale, par)
    if cp_conjugate:
        wc = conjugate_wc(wc)
    wc_eff = get_wceff(q2, wc, par, B, V, lep, scale)
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mV = par['m_'+V]
    mb = running.get_mb(par, scale)
    N = prefactor(q2, par, B, V)
    h = angular.helicity_amps_v(q2, mB, mV, mb, 0, ml, ml, ff, wc_eff, N)
    return h


# get spectator scattering contribution
def get_ss(q2, wc_obj, par_dict, B, V, cp_conjugate):
    # this only needs to be done for low q2 - which doesn't exist for taus!
    if q2 >= 8.9:
        return {('0' ,'V'): 0, ('pl' ,'V'): 0, ('mi' ,'V'): 0, }
    ss_name = B+'->'+V+'ll spectator scattering'
    return AuxiliaryQuantity[ss_name].prediction(par_dict=par_dict, wc_obj=wc_obj, q2=q2, cp_conjugate=cp_conjugate)

# get subleading hadronic contribution at low q2
def get_subleading(q2, wc_obj, par_dict, B, V, cp_conjugate):
    if q2 <= 9:
        sub_name = B+'->'+V+ 'll subleading effects at low q2'
        return AuxiliaryQuantity[sub_name].prediction(par_dict=par_dict, wc_obj=wc_obj, q2=q2, cp_conjugate=cp_conjugate)
    elif q2 > 14:
        sub_name = B+'->'+V+ 'll subleading effects at high q2'
        return AuxiliaryQuantity[sub_name].prediction(par_dict=par_dict, wc_obj=wc_obj, q2=q2, cp_conjugate=cp_conjugate)
    else:
        return {}

def helicity_amps(q2, ff, wc_obj, par, B, V, lep):
    if q2 >= 8.7 and q2 < 14:
        warnings.warn("The predictions in the region of narrow charmonium resonances are not meaningful")
    return add_dict((
        helicity_amps_ff(q2, ff, wc_obj, par, B, V, lep, cp_conjugate=False),
        get_ss(q2, wc_obj, par, B, V, cp_conjugate=False),
        get_subleading(q2, wc_obj, par, B, V, cp_conjugate=False)
        ))

def helicity_amps_bar(q2, ff, wc_obj, par, B, V, lep):
    if q2 >= 8.7 and q2 < 14:
        warnings.warn("The predictions in the region of narrow charmonium resonances are not meaningful")
    return add_dict((
        helicity_amps_ff(q2, ff, wc_obj, par, B, V, lep, cp_conjugate=True),
        get_ss(q2, wc_obj, par, B, V, cp_conjugate=True),
        get_subleading(q2, wc_obj, par, B, V, cp_conjugate=True)
        ))
