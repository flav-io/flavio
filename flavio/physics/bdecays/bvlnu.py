r"""Functions for exclusive $B\to V\ell\nu$ decays."""

from math import sqrt,pi
import flavio
from flavio.physics.bdecays.common import lambda_K, meson_quark, meson_ff
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict
from flavio.physics import ckm
from flavio.classes import AuxiliaryQuantity
from flavio.config import config
from flavio.physics.running import running
from flavio.physics.bdecays import angular
from flavio.physics.bdecays.wilsoncoefficients import get_wceff_fccc
from flavio.classes import Observable, Prediction


def get_ff(q2, par, B, V):
    ff_name = meson_ff[(B,V)] + ' form factor'
    return AuxiliaryQuantity.get_instance(ff_name).prediction(par_dict=par, wc_obj=None, q2=q2)

def prefactor(q2, par, B, V, lep):
    GF = par['GF']
    scale = config['renormalization scale']['bvll']
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mV = par['m_'+V]
    tauB = par['tau_'+B]
    laB  = lambda_K(mB**2, mV**2, q2)
    laGa = lambda_K(q2, ml**2, 0.)
    qi_qj = meson_quark[(B, V)]
    if qi_qj == 'bu':
        Vij = ckm.get_ckm(par)[0,2] # V_{ub} for b->u transitions
    if qi_qj == 'bc':
        Vij = ckm.get_ckm(par)[1,2] # V_{cb} for b->c transitions
    if q2 <= ml**2:
        return 0
    return 4*GF/sqrt(2)*Vij

def get_angularcoeff(q2, wc_obj, par, B, V, lep):
    scale = config['renormalization scale']['bvll']
    mb = running.get_mb(par, scale)
    wc = get_wceff_fccc(wc_obj, par, meson_quark[(B,V)], lep, mb, scale, nf=5)
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mV = par['m_'+V]
    qi_qj = meson_quark[(B, V)]
    if qi_qj == 'bu':
        mlight = 0. # neglecting the up quark mass
    if qi_qj == 'bc':
        mlight = running.get_mc(par, scale) # this is needed for scalar contributions
    N = prefactor(q2, par, B, V, lep)
    ff = get_ff(q2, par, B, V)
    h = angular.helicity_amps_v(q2, mB, mV, mb, mlight, ml, 0, ff, wc, N)
    J = angular.angularcoeffs_general_v(h, q2, mB, mV, mb, mlight, ml, 0)
    return J

def dGdq2(J):
    return 3/4. * (2 * J['1s'] + J['1c']) - 1/4. * (2 * J['2s'] + J['2c'])

def dBRdq2_lep(q2, wc_obj, par, B, V, lep):
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mV = par['m_'+V]
    if q2 < ml**2 or q2 > (mB-mV)**2:
        return 0
    tauB = par['tau_'+B]
    J = get_angularcoeff(q2, wc_obj, par, B, V, lep)
    if V == 'rho0' or V == 'omega':
        # factor of 1/2 for neutral rho due to rho = (uubar-ddbar)/sqrt(2)
        # and also for omega = (uubar+ddbar)/sqrt(2)
        return tauB * dGdq2(J) / 2.
    return tauB * dGdq2(J)


def dBRdq2(q2, wc_obj, par, B, V, lep):
    if lep == 'l':
        # average of e and mu!
        return (dBRdq2_lep(q2, wc_obj, par, B, V, 'e') + dBRdq2_lep(q2, wc_obj, par, B, V, 'mu'))/2
    else:
        return dBRdq2_lep(q2, wc_obj, par, B, V, lep)



def dBRdq2_function(B, V, lep):
    return lambda wc_obj, par, q2: dBRdq2(q2, wc_obj, par, B, V, lep)

def BR_binned(q2min, q2max, wc_obj, par, B, V, lep):
    def integrand(q2):
        return dBRdq2(q2, wc_obj, par, B, V, lep)
    return flavio.math.integrate.nintegrate(integrand, q2min, q2max)

def BR_binned_function(B, V, lep):
    return lambda wc_obj, par, q2min, q2max: BR_binned(q2min, q2max, wc_obj, par, B, V, lep)

def _BR_tot(wc_obj, par, B, V, lep):
    mB = par['m_'+B]
    mV = par['m_'+V]
    ml = par['m_'+lep]
    q2max = (mB-mV)**2
    q2min = ml**2
    return BR_binned(q2min, q2max, wc_obj, par, B, V, lep)

def BR_tot(wc_obj, par, B, V, lep):
    if lep == 'l':
        # average of e and mu!
        return (_BR_tot(wc_obj, par, B, V, 'e')+_BR_tot(wc_obj, par, B, V, 'mu'))/2.
    else:
        return _BR_tot(wc_obj, par, B, V, lep)

def BR_tot_function(B, V, lep):
    return lambda wc_obj, par: BR_tot(wc_obj, par, B, V, lep)


# Observable and Prediction instances

_tex = {'e': 'e', 'mu': '\mu', 'tau': r'\tau', 'l': r'\ell'}
_func = {'dBR/dq2': dBRdq2_function, 'BR': BR_tot_function, '<BR>': BR_binned_function}
_desc = {'dBR/dq2': 'Differential', 'BR': 'Total', '<BR>': 'Binned'}
_tex_br = {'dBR/dq2': r'\frac{d\text{BR}}{dq^2}', 'BR': r'\text{BR}', '<BR>': r'\langle\text{BR}\rangle'}
_args = {'dBR/dq2': ['q2'], 'BR': None, '<BR>': ['q2min', 'q2max']}
_hadr = {
'B0->D*': {'tex': r"B^0\to D^{\ast -}", 'B': 'B0', 'V': 'D*+', },
'B+->D*': {'tex': r"B^+\to D^{\ast 0}", 'B': 'B+', 'V': 'D*0', },
'B0->rho': {'tex': r"B^0\to \rho^-", 'B': 'B0', 'V': 'rho+', },
'B+->rho': {'tex': r"B^+\to \rho^0", 'B': 'B+', 'V': 'rho0', },
'B+->omega': {'tex': r"B^+\to \omega ", 'B': 'B+', 'V': 'omega', },
'Bs->K*': {'tex': r"B_s\to K^{* -} ", 'B': 'Bs', 'V': 'K*+', },
}
for l in ['e', 'mu', 'tau', 'l']:
    for br in ['dBR/dq2', 'BR', '<BR>']:
        for M in _hadr.keys():
            _obs_name = br + "("+M+l+"nu)"
            _obs = Observable(_obs_name)
            _obs.set_description(_desc[br] + r" branching ratio of $"+_hadr[M]['tex']+_tex[l]+r"^+\nu_"+_tex[l]+"$")
            _obs.tex = r'$' + _tex_br[br] + r"("+_hadr[M]['tex']+_tex[l]+r"^+\nu_"+_tex[l]+")$"
            _obs.arguments = _args[br]
            Prediction(_obs_name, _func[br](_hadr[M]['B'], _hadr[M]['V'], l))
