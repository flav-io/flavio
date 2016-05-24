r"""Observables in $B\to V\nu\bar\nu$"""

import flavio
import scipy.integrate
from flavio.classes import Observable, Prediction
from math import sqrt, pi


def prefactor(q2, par, B, V):
    GF = par['GF']
    scale = flavio.config['renormalization scale']['bvll']
    alphaem = flavio.physics.running.running.get_alpha(par, scale)['alpha_e']
    di_dj = flavio.physics.bdecays.common.meson_quark[(B,V)]
    xi_t = flavio.physics.ckm.xi('t',di_dj)(par)
    return 4*GF/sqrt(2)*xi_t*alphaem/(4*pi)

def get_ff(q2, par, B, V):
    ff_name = flavio.physics.bdecays.common.meson_ff[(B,V)] + ' form factor'
    return flavio.classes.AuxiliaryQuantity.get_instance(ff_name).prediction(par_dict=par, wc_obj=None, q2=q2)

def helicity_amps(q2, wc_obj, par, B, V, nu1, nu2):
    scale = flavio.config['renormalization scale']['bvll']
    label = flavio.physics.bdecays.common.meson_quark[(B,V)] + nu1 + nu2 # e.g. bsnuenue, bdnutaunumu
    wc = wc_obj.get_wc(label, scale, par)
    if nu1 == nu2: # add the SM contribution if neutrino flavours coincide
        wc['CL_'+label] += flavio.physics.bdecays.wilsoncoefficients.CL_SM(par)
    mB = par['m_'+B]
    mV = par['m_'+V]
    N = prefactor(q2, par, B, V)
    ff = get_ff(q2, par, B, V)
    wc_eff = flavio.physics.bdecays.wilsoncoefficients.get_wceff_nunu(q2, wc, par, B, V, nu1, nu2, scale)
    # below, mb is set to 4 just to save time. It doesn't enter at all as there
    # are no dipole operators.
    h = flavio.physics.bdecays.angular.helicity_amps_v(q2, mB, mV, 4, 0, 0, 0, ff, wc_eff, N)
    return h

def bvnunu_obs(function, q2, wc_obj, par, B, V, nu1, nu2):
    mB = par['m_'+B]
    mV = par['m_'+V]
    h = helicity_amps(q2, wc_obj, par, B, V, nu1, nu2)
    # below, mb is set to 4 just to save time. It doesn't enter at all as there
    # are no dipole operators.
    J = flavio.physics.bdecays.angular.angularcoeffs_general_v(h, q2, mB, mV, 4, 0, 0, 0)
    return function(J)

def bvnunu_dbrdq2_summed(q2, wc_obj, par, B, V):
    tauB = par['tau_'+B]
    lep =  ['e', 'mu', 'tau']
    dbrdq2 = tauB * sum([ bvnunu_obs(flavio.physics.bdecays.bvll.observables.dGdq2, q2, wc_obj, par, B, V, 'nu'+nu1, 'nu'+nu2)
        for nu1 in lep for nu2 in lep ])
    if V == 'rho0':
        # factor of 1/2 for neutral rho due to rho0 = (uubar-ddbar)/sqrt(2)
        return dbrdq2/2.
    else:
        return dbrdq2

def bvnunu_dbrdq2_int_summed(q2min, q2max, wc_obj, par, B, V):
    def obs(q2):
        return bvnunu_dbrdq2_summed(q2, wc_obj, par, B, V)
    return flavio.math.integrate.nintegrate(obs, q2min, q2max)/(q2max-q2min)

def bvnunu_BRtot_summed(wc_obj, par, B, V):
    def obs(q2):
        return bvnunu_dbrdq2_summed(q2, wc_obj, par, B, V)
    mB = par['m_'+B]
    mV = par['m_'+V]
    q2max = (mB-mV)**2
    q2min = 0
    return flavio.math.integrate.nintegrate(obs, q2min, q2max)

# Functions returning functions needed for Prediction instances

def BRtot_summed(B, V):
    return lambda wc_obj, par: bvnunu_BRtot_summed(wc_obj, par, B, V)

def dbrdq2_int_summed(B, V):
    def fct(wc_obj, par, q2min, q2max):
        return bvnunu_dbrdq2_int_summed(q2min, q2max, wc_obj, par, B, V)
    return fct

def dbrdq2_summed(B, V):
    def fct(wc_obj, par, q2):
        return bvnunu_dbrdq2_summed(q2, wc_obj, par, B, V)
    return fct


_hadr = {
'B0->K*': {'tex': r"B^0\to K^{*0}", 'B': 'B0', 'V': 'K*0', },
'B+->K*': {'tex': r"B^+\to K^{*+}", 'B': 'B+', 'V': 'K*+', },
'B+->rho': {'tex': r"B^+\to \rho^{+}", 'B': 'B+', 'V': 'rho+', },
'B0->rho': {'tex': r"B^0\to \rho^{0}", 'B': 'B0', 'V': 'rho0', },
}

for M in _hadr.keys():

    # binned branching ratio
    _obs_name = "<dBR/dq2>("+M+"nunu)"
    _obs = flavio.classes.Observable(name=_obs_name, arguments=['q2min', 'q2max'])
    _obs.set_description(r"Binned differential branching ratio of $" + _hadr[M]['tex'] + r"\nu\bar\nu$")
    _obs.tex = r"$\langle \frac{d\text{BR}}{dq^2} \rangle(" + _hadr[M]['tex'] + r"\nu\bar\nu)$"
    flavio.classes.Prediction(_obs_name, dbrdq2_int_summed(_hadr[M]['B'], _hadr[M]['V']))

    # differential branching ratio
    _obs_name = "dBR/dq2("+M+"nunu)"
    _obs = flavio.classes.Observable(name=_obs_name, arguments=['q2'])
    _obs.set_description(r"Differential branching ratio of $" + _hadr[M]['tex'] + r"\nu\bar\nu$")
    _obs.tex = r"$\frac{d\text{BR}}{dq^2}(" + _hadr[M]['tex'] + r"\nu\bar\nu)$"
    flavio.classes.Prediction(_obs_name, dbrdq2_summed(_hadr[M]['B'], _hadr[M]['V']))

    # total branching ratio
    _obs_name = "BR("+M+"nunu)"
    _obs = flavio.classes.Observable(name=_obs_name)
    _obs.set_description(r"Branching ratio of $" + _hadr[M]['tex'] + r"\nu\bar\nu$")
    _obs.tex = r"$\text{BR}(" + _hadr[M]['tex'] + r"\nu\bar\nu)$"
    flavio.classes.Prediction(_obs_name, BRtot_summed(_hadr[M]['B'], _hadr[M]['V']))
