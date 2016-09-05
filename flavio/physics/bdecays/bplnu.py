r"""Functions for exclusive $B\to P\ell\nu$ decays."""

from math import sqrt
import flavio
from flavio.physics.bdecays.common import meson_quark, meson_ff
from flavio.physics import ckm
from flavio.classes import AuxiliaryQuantity
from flavio.config import config
from flavio.physics.running import running
from flavio.physics.bdecays import angular
from flavio.physics.bdecays.wilsoncoefficients import get_wceff_fccc
from flavio.classes import Observable, Prediction



def prefactor(q2, par, B, P, lep):
    GF = par['GF']
    ml = par['m_'+lep]
    scale = config['renormalization scale']['bpll']
    alphaem = running.get_alpha(par, scale)['alpha_e']
    di_dj = meson_quark[(B,P)]
    qi_qj = meson_quark[(B, P)]
    if qi_qj == 'bu':
        Vij = ckm.get_ckm(par)[0,2] # V_{ub} for b->u transitions
    if qi_qj == 'bc':
        Vij = ckm.get_ckm(par)[1,2] # V_{cb} for b->c transitions
    if q2 <= ml**2:
        return 0
    return 4*GF/sqrt(2)*Vij

def get_ff(q2, par, B, P):
    ff_name = meson_ff[(B,P)] + ' form factor'
    return AuxiliaryQuantity.get_instance(ff_name).prediction(par_dict=par, wc_obj=None, q2=q2)

def get_angularcoeff(q2, wc_obj, par, B, P, lep):
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mP = par['m_'+P]
    scale = config['renormalization scale']['bpll']
    mb = running.get_mb(par, scale)
    wc = get_wceff_fccc(wc_obj, par, meson_quark[(B,P)], lep, mb, scale, nf=5)
    N = prefactor(q2, par, B, P, lep)
    ff = get_ff(q2, par, B, P)
    qi_qj = meson_quark[(B, P)]
    if qi_qj == 'bu':
        mlight = 0. # neglecting the up quark mass
    if qi_qj == 'bc':
        mlight = running.get_mc(par, scale) # this is needed for scalar contributions
    h = angular.helicity_amps_p(q2, mB, mP, mb, mlight, ml, 0, ff, wc, N)
    J = angular.angularcoeffs_general_p(h, q2, mB, mP, mb, mlight, ml, 0)
    return J

def dGdq2(J):
    return 2 * (J['a'] + J['c']/3.)

def dBRdq2_lep(q2, wc_obj, par, B, P, lep):
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mP = par['m_'+P]
    if q2 < ml**2 or q2 > (mB-mP)**2:
        return 0
    tauB = par['tau_'+B]
    J = get_angularcoeff(q2, wc_obj, par, B, P, lep)
    if P == 'pi0':
        # factor of 1/2 for neutral pi due to pi = (uubar-ddbar)/sqrt(2)
        return tauB * dGdq2(J) / 2.
    return tauB * dGdq2(J)

def dBRdq2(q2, wc_obj, par, B, P, lep):
    if lep == 'l':
        # average of e and mu!
        return (dBRdq2_lep(q2, wc_obj, par, B, P, 'e') + dBRdq2_lep(q2, wc_obj, par, B, P, 'mu'))/2
    else:
        return dBRdq2_lep(q2, wc_obj, par, B, P, lep)


def dBRdq2_function(B, P, lep):
    return lambda wc_obj, par, q2: dBRdq2(q2, wc_obj, par, B, P, lep)

def BR_binned(q2min, q2max, wc_obj, par, B, P, lep):
    def integrand(q2):
        return dBRdq2(q2, wc_obj, par, B, P, lep)
    return flavio.math.integrate.nintegrate(integrand, q2min, q2max)

def BR_binned_function(B, P, lep):
    return lambda wc_obj, par, q2min, q2max: BR_binned(q2min, q2max, wc_obj, par, B, P, lep)

def _BR_tot(wc_obj, par, B, P, lep):
    mB = par['m_'+B]
    mP = par['m_'+P]
    ml = par['m_'+lep]
    q2max = (mB-mP)**2
    q2min = ml**2
    return BR_binned(q2min, q2max, wc_obj, par, B, P, lep)

def BR_tot(wc_obj, par, B, P, lep):
    if lep == 'l':
        # average of e and mu!
        return (_BR_tot(wc_obj, par, B, P, 'e')+_BR_tot(wc_obj, par, B, P, 'mu'))/2.
    else:
        return _BR_tot(wc_obj, par, B, P, lep)


def BR_tot_function(B, P, lep):
    return lambda wc_obj, par: BR_tot(wc_obj, par, B, P, lep)


# Observable and Prediction instances

_tex = {'e': 'e', 'mu': '\mu', 'tau': r'\tau', 'l': r'\ell'}
_func = {'dBR/dq2': dBRdq2_function, 'BR': BR_tot_function, '<BR>': BR_binned_function}
_desc = {'dBR/dq2': 'Differential', 'BR': 'Total', '<BR>': 'Binned'}
_tex_br = {'dBR/dq2': r'\frac{d\text{BR}}{dq^2}', 'BR': r'\text{BR}', '<BR>': r'\langle\text{BR}\rangle'}
_args = {'dBR/dq2': ['q2'], 'BR': None, '<BR>': ['q2min', 'q2max']}

for l in ['e', 'mu', 'tau', 'l']:
    for br in ['dBR/dq2', 'BR', '<BR>']:
        _obs_name = br + "(B+->D"+l+"nu)"
        _obs = Observable(_obs_name)
        _obs.set_description(_desc[br] + r" branching ratio of $B^+\to D^{0}"+_tex[l]+r"^+\nu_"+_tex[l]+"$")
        _obs.tex = r'$' + _tex_br[br] + r"(B^+\to D^{0}"+_tex[l]+r"^+\nu_"+_tex[l]+")$"
        _obs.arguments = _args[br]
        Prediction(_obs_name, _func[br]('B+', 'D0', l))

        _obs_name = br + "(B0->D"+l+"nu)"
        _obs = Observable(_obs_name)
        _obs.set_description(_desc[br] + r" branching ratio of $B^0\to D^{-}"+_tex[l]+r"^+\nu_"+_tex[l]+"$")
        _obs.tex = r'$' + _tex_br[br] + r"(B^0\to D^{-}"+_tex[l]+r"^+\nu_"+_tex[l]+")$"
        _obs.arguments = _args[br]
        Prediction(_obs_name, _func[br]('B0', 'D+', l))

        # _obs_name = br + "(Bs->K"+l+"nu)"
        # _obs = Observable(_obs_name)
        # _obs.set_description(_desc[br] + r" branching ratio of $B_s\to K^{-}"+_tex[l]+r"^+\nu_"+_tex[l]+"$")
        # _obs.tex = r'$' + _tex_br[br] + r"(B_s\to K^{-}"+_tex[l]+r"^+\nu_"+_tex[l]+")$"
        # _obs.arguments = _args[br]
        # Prediction(_obs_name, _func[br]('Bs', 'K+', l))

        _obs_name = br + "(B+->pi"+l+"nu)"
        _obs = Observable(_obs_name)
        _obs.set_description(_desc[br] + r" branching ratio of $B^+\to \pi^0"+_tex[l]+r"^+\nu_"+_tex[l]+"$")
        _obs.tex = r'$' + _tex_br[br] + r"(B^+\to \pi^0"+_tex[l]+r"^+\nu_"+_tex[l]+")$"
        _obs.arguments = _args[br]
        Prediction(_obs_name, _func[br]('B+', 'pi0', l))

        _obs_name = br + "(B0->pi"+l+"nu)"
        _obs = Observable(_obs_name)
        _obs.set_description(_desc[br] + r" branching ratio of $B^0\to \pi^-"+_tex[l]+r"^+\nu_"+_tex[l]+"$")
        _obs.tex = r'$' + _tex_br[br] + r"(B^0\to \pi^-"+_tex[l]+r"^+\nu_"+_tex[l]+")$"
        _obs.arguments = _args[br]
        Prediction(_obs_name, _func[br]('B0', 'pi+', l))
