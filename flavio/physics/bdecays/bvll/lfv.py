r"""Observables in lepton flavour violating decays $B\to V\ell\ell^\prime$"""

import flavio
import scipy.integrate
from flavio.classes import Observable, Prediction
from math import sqrt, pi


def prefactor(q2, par, B, V, l1, l2):
    GF = par['GF']
    ml1 = par['m_'+l1]
    ml2 = par['m_'+l2]
    scale = flavio.config['renormalization scale']['bvll']
    alphaem = flavio.physics.running.running.get_alpha(par, scale)['alpha_e']
    di_dj = flavio.physics.bdecays.common.meson_quark[(B,V)]
    xi_t = flavio.physics.ckm.xi('t',di_dj)(par)
    if q2 <= (ml1+ml2)**2:
        return 0
    return 4*GF/sqrt(2)*xi_t*alphaem/(4*pi)

def helicity_amps(q2, wc, par, B, V, l1, l2):
    scale = flavio.config['renormalization scale']['bvll']
    wc_eff = flavio.physics.bdecays.wilsoncoefficients.get_wceff_lfv(q2, wc, par, B, V, l1, l2, scale)
    ml1 = par['m_'+l1]
    ml2 = par['m_'+l2]
    mB = par['m_'+B]
    mV = par['m_'+V]
    mb = flavio.physics.running.running.get_mb(par, scale)
    N = prefactor(q2, par, B, V, l1, l2)
    ff = flavio.physics.bdecays.bvll.amplitudes.get_ff(q2, par, B, V)
    h = flavio.physics.bdecays.angular.helicity_amps_v(q2, mB, mV, mb, 0, ml1, ml2, ff, wc_eff, N)
    return h

def bvlilj_obs(function, q2, wc, par, B, V, l1, l2):
    ml1 = par['m_'+l1]
    ml2 = par['m_'+l2]
    mB = par['m_'+B]
    mV = par['m_'+V]
    if q2 < (ml1+ml2)**2 or q2 > (mB-mV)**2:
        return 0
    scale = flavio.config['renormalization scale']['bvll']
    mb = flavio.physics.running.running.get_mb(par, scale)
    h = helicity_amps(q2, wc, par, B, V, l1, l2)
    J = flavio.physics.bdecays.angular.angularcoeffs_general_v(h, q2, mB, mV, mb, 0, ml1, ml2)
    return function(J)

def bvlilj_obs_int(function, q2min, q2max, wc, par, B, V, l1, l2):
    def obs(q2):
        return bvlilj_obs(function, q2, wc, par, B, V, l1, l2)
    return flavio.math.integrate.nintegrate(obs, q2min, q2max)

def BR_tot(wc_obj, par, B, V, l1, l2):
    scale = flavio.config['renormalization scale']['bvll']
    label = flavio.physics.bdecays.common.meson_quark[(B,V)] + l1 + l2 # e.g. bsemu, bdtaue
    wc = wc_obj.get_wc(label, scale, par)
    if all([abs(v) < 1e-12 for v in wc.values()]):
        # if all WCs are essentially zero, return BR=0
        return 0
    mB = par['m_'+B]
    mV = par['m_'+V]
    ml1 = par['m_'+l1]
    ml2 = par['m_'+l2]
    q2max = (mB-mV)**2
    q2min = (ml1+ml2)**2
    tauB = par['tau_'+B]
    return tauB*bvlilj_obs_int(flavio.physics.bdecays.bvll.observables.dGdq2,
                          q2min, q2max, wc, par, B, V, l1, l2)

def BR_tot_function(B, V, l1, l2):
    return lambda wc_obj, par: BR_tot(wc_obj, par, B, V, l1, l2)

def BR_tot_function_leptonsum(B, V, l1, l2):
    return lambda wc_obj, par: BR_tot(wc_obj, par, B, V, l1, l2) + BR_tot(wc_obj, par, B, V, l2, l1)



# Observable and Prediction instances

_tex = {'emu': r'e^+\mu^-', 'mue': r'\mu^+e^-',
    'taue': r'\tau^+e^-', 'etau': r'e^+\tau^-',
    'taumu': r'\tau^+\mu^-', 'mutau': r'\mu^+\tau^-'}
_tex_lsum = {'emu': r'e^\pm\mu^\mp',  'etau': r'e^\pm\tau^\mp', 'mutau': r'\mu^\pm\tau^\mp'}
_func = {'BR': BR_tot_function, }
_desc = { 'BR': 'Total', }
_tex_br = {'BR': r'\text{BR}', }
_args = {'BR': None, }
_hadr = {
'B0->K*': {'tex': r"\bar B^0\to \bar K^{*0}", 'B': 'B0', 'V': 'K*0', },
'B+->K*': {'tex': r"B^-\to K^{*-}", 'B': 'B+', 'V': 'K*+', },
'B+->rho': {'tex': r"B^-\to \rho^{-}", 'B': 'B+', 'V': 'rho+', },
'B0->rho': {'tex': r"\bar B^0\to \rho^{0}", 'B': 'B0', 'V': 'rho0', },
'Bs->phi': {'tex': r"\bar B_s\to \phi", 'B': 'Bs', 'V': 'phi', },
}

for ll in [('e','mu'), ('mu','e'), ('e','tau'), ('tau','e'), ('mu','tau'), ('tau','mu')]:
    for M in _hadr.keys():

        _process_tex = _hadr[M]['tex']+' '+_tex[''.join(ll)]
        _process_taxonomy = r'Process :: $b$ hadron decays :: FCNC decays :: $B\to V\ell^+\ell^-$ :: $' + _process_tex + r"$"

        for br in ['BR',]:
            _obs_name = br + "("+M+''.join(ll)+")"
            _obs = Observable(_obs_name)
            _obs.set_description(_desc[br] + r" branching ratio of $"+_process_tex+r"$")
            _obs.tex = r'$' + _tex_br[br] + "(" + _process_tex+r")$"
            _obs.arguments = _args[br]
            _obs.add_taxonomy(_process_taxonomy)
            Prediction(_obs_name, _func[br](_hadr[M]['B'], _hadr[M]['V'], ll[0], ll[1]))
