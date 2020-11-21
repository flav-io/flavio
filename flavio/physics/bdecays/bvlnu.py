r"""Functions for exclusive $B\to V\ell\nu$ decays."""

from math import sqrt, pi, cos, sin
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
    """Return the form factors"""
    ff_name = meson_ff[(B,V)] + ' form factor'
    return AuxiliaryQuantity[ff_name].prediction(par_dict=par, wc_obj=None, q2=q2)

def prefactor(q2, par, B, V, lep):
    """Return the prefactor including constants and CKM elements"""
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
    Jlist = [_get_angularcoeff(q2, wc_obj, par, B, V, lep, nu)
             for nu in ['e', 'mu', 'tau']]
    J = {}
    J['1s'] = sum([JJ['1s'] for JJ in Jlist])
    J['1c'] = sum([JJ['1c'] for JJ in Jlist])
    J['2s'] = sum([JJ['2s'] for JJ in Jlist])
    J['2c'] = sum([JJ['2c'] for JJ in Jlist])
    J['6s'] = sum([JJ['6s'] for JJ in Jlist])
    J['6c'] = sum([JJ['6c'] for JJ in Jlist])
    J[3] = sum([JJ[3] for JJ in Jlist])
    J[4] = sum([JJ[4] for JJ in Jlist])
    J[5] = sum([JJ[5] for JJ in Jlist])
    J[7] = sum([JJ[7] for JJ in Jlist])
    J[8] = sum([JJ[8] for JJ in Jlist])
    J[9] = sum([JJ[9] for JJ in Jlist])
    return J



def _get_angularcoeff(q2, wc_obj, par, B, V, lep, nu):
    scale = config['renormalization scale']['bvll']
    mb = running.get_mb(par, scale)
    wc = get_wceff_fccc(wc_obj, par, meson_quark[(B,V)], lep, nu, mb, scale, nf=5)
    if lep != nu and all(C == 0 for C in wc.values()):
        # if all WCs vanish, so does the AC!
        return {k: 0 for k in
                ['1s', '1c', '2s', '2c', '6s', '6c', 3, 4, 5, 7, 8, 9]}
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
    r"""$q^2$-differential branching ratio in terms of angular coefficients."""
    return 3/4. * (2 * J['1s'] + J['1c']) - 1/4. * (2 * J['2s'] + J['2c'])

def dGdq2_L(J):
    r"""$q^2$-differential branching ratio to longitudinally polarized
    vector meson in terms of angular coefficients."""
    return 3/4. * J['1c'] - 1/4. * J['2c']

def dGdq2_T(J):
    r"""$q^2$-differential branching ratio to transversely polarized
    vector meson in terms of angular coefficients."""
    return 3/2. * J['1s'] - 1/2. * J['2s']

# For the angle-differential and binned distributions, the main idea is this:
# while the q2-integration has to be done numerically, the angle integration is
# trivial to do analytically as the angular dependence is given in terms of
# trigonometric functions. So the differential distributions are given as
# dictionaries with (q2-dependent) coefficients of these angular functions.
# Integration (i.e. binning) in angles then merely amounts to replacing the
# angular functions by their respective integrals.

def dG_dq2_dcosthl(J):
    r"""$\cos\theta_\ell$-differential branching ratio in terms of angular
    coefficients, as dictionary of coefficients of trigonometric functions
    of $\theta_\ell$."""
    return {'1': 3/8. * (J['1c'] + 2*J['1s']),
            'c': 3/8. * (J['6c'] + 2*J['6s']),
            'c2': 3/8. * (J['2c'] + 2*J['2s']) }

def dG_dq2_dcosthV(J):
    r"""$\cos\theta_V$-differential branching ratio in terms of angular
    coefficients, as dictionary of coefficients of trigonometric functions
    of $\theta_V$."""
    return {'c^2': -3/8. * (-3*J['1c'] + J['2c']),
            's^2': -3/8. * (-3*J['1s'] + J['2s']) }

def dG_dq2_dphi(J):
    r"""$\phi$-differential branching ratio in terms of angular
    coefficients, as dictionary of coefficients of trigonometric functions
    of $\phi$."""
    return {'1': 1/(8*pi) * (3*J['1c'] + 6*J['1s'] - J['2c'] - 2*J['2s']),
            'c2': 1/(2*pi) * J[3],
            's2': 1/(2*pi) * J[9] }

def _cos_angle_diff(costh):
    r"""Trigonometric functions for differential distributions in terms of
    $\cos\theta_{\ell,V}$"""
    return {'1': 1, 'c': costh, 'c2': 2*costh**2-1, 'c^2': costh**2,
            's^2': 1 - costh**2, 's2': 2*costh*sqrt(1-costh**2)}

def _cos_angle_int(costh):
    r"""Integrated trigonometric functions for binned distributions in terms of
    $\cos\theta_{\ell,V}$"""
    return {'1': costh, 'c': costh**2/2., 'c2': 2*costh**3/3.-costh,
            'c^2': costh**3/3., 's^2': costh - costh**3/3.,
            's2': -2/3.*(1-costh**2)**(3/2.)}

def _angle_diff(phi):
    r"""Trigonometric functions for differential distributions in terms of
    $\phi$"""
    return {'1': 1, 'c2': cos(2*phi), 's2': sin(2*phi)}

def _angle_int(phi):
    r"""Integrated trigonometric functions for binned distributions in terms of
    $\phi$"""
    return {'1': phi, 'c2': sin(2*phi)/2., 's2': -cos(2*phi)/2.}

def obs_q2int(fct, wc_obj, par, B, V, lep):
    """q2-integrated observable"""
    mB = par['m_'+B]
    mV = par['m_'+V]
    ml = par['m_'+lep]
    q2max = (mB-mV)**2
    q2min = ml**2
    def integrand(q2):
        return fct(q2)
    return flavio.math.integrate.nintegrate(integrand, q2min, q2max)

def kinem_allowed(q2, par, B, V, lep):
    """True if q2 is in the kinematically allowed region"""
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mV = par['m_'+V]
    if q2 < ml**2 or q2 > (mB-mV)**2:
        return False
    else:
        return True

def FL_diff(q2, wc_obj, par, B, V, lep):
    if not kinem_allowed(q2, par, B, V, lep):
        return 0
    J = get_angularcoeff(q2, wc_obj, par, B, V, lep)
    return dGdq2_L(J) / dGdq2(J)


def FL_binned(q2min, q2max, wc_obj, par, B, V, lep):
    num = flavio.math.integrate.nintegrate(lambda q2: dBRdq2(q2, wc_obj, par, B, V, lep, A='L'), q2min, q2max)
    if num == 0:
        return 0
    denom = flavio.math.integrate.nintegrate(lambda q2: dBRdq2(q2, wc_obj, par, B, V, lep,  A=None), q2min, q2max)
    return num / denom

def Itot_norm(fct_J, wc_obj, par, B, V, lep):
    def fct(q2):
        J = get_angularcoeff(q2, wc_obj, par, B, V, lep)
        return fct_J(J)
    num = obs_q2int(fct, wc_obj, par, B, V, lep)
    def fct_den(q2):
        J = get_angularcoeff(q2, wc_obj, par, B, V, lep)
        return dGdq2(J)
    den = obs_q2int(fct_den, wc_obj, par, B, V, lep)
    return num / den

def dBR_dq2_dcosthl_binned(q2, clmin, clmax, wc_obj, par, B, V, lep):
    if not kinem_allowed(q2, par, B, V, lep):
        return 0
    tauB = par['tau_'+B]
    J = get_angularcoeff(q2, wc_obj, par, B, V, lep)
    dG = dG_dq2_dcosthl(J)
    ang_min = _cos_angle_int(clmin)
    ang_max = _cos_angle_int(clmax)
    return BRfac(V) * tauB * sum(
                        [y * (ang_max[a] - ang_min[a]) for a, y in dG.items()])

def BR_binned_costhl(clmin, clmax, wc_obj, par, B, V, lep):
    def fct(q2):
        return dBR_dq2_dcosthl_binned(q2, clmin, clmax, wc_obj, par, B, V, lep)
    return obs_q2int(fct, wc_obj, par, B, V, lep)

def dBR_dq2_dcosthl(q2, cl, wc_obj, par, B, V, lep):
    if not kinem_allowed(q2, par, B, V, lep):
        return 0
    tauB = par['tau_'+B]
    J = get_angularcoeff(q2, wc_obj, par, B, V, lep)
    dG = dG_dq2_dcosthl(J)
    ang = _cos_angle_diff(cl)
    return BRfac(V) * tauB * sum(
                        [y * ang[a] for a, y in dG.items()])

def dBR_dcosthl(cl, wc_obj, par, B, V, lep):
    def fct(q2):
        return dBR_dq2_dcosthl(q2, cl, wc_obj, par, B, V, lep)
    return obs_q2int(fct, wc_obj, par, B, V, lep)

def dBR_dq2_dcosthV_binned(q2, cVmin, cVmax, wc_obj, par, B, V, lep):
    if not kinem_allowed(q2, par, B, V, lep):
        return 0
    tauB = par['tau_'+B]
    J = get_angularcoeff(q2, wc_obj, par, B, V, lep)
    dG = dG_dq2_dcosthV(J)
    ang_min = _cos_angle_int(cVmin)
    ang_max = _cos_angle_int(cVmax)
    return BRfac(V) * tauB * sum(
                        [y * (ang_max[a] - ang_min[a]) for a, y in dG.items()])

def BR_binned_costhV(cVmin, cVmax, wc_obj, par, B, V, lep):
    def fct(q2):
        return dBR_dq2_dcosthV_binned(q2, cVmin, cVmax, wc_obj, par, B, V, lep)
    return obs_q2int(fct, wc_obj, par, B, V, lep)

def dBR_dq2_dcosthV(q2, cV, wc_obj, par, B, V, lep):
    if not kinem_allowed(q2, par, B, V, lep):
        return 0
    tauB = par['tau_'+B]
    J = get_angularcoeff(q2, wc_obj, par, B, V, lep)
    dG = dG_dq2_dcosthV(J)
    ang = _cos_angle_diff(cV)
    return BRfac(V) * tauB * sum(
                        [y * ang[a] for a, y in dG.items()])

def dBR_dcosthV(cV, wc_obj, par, B, V, lep):
    def fct(q2):
        return dBR_dq2_dcosthV(q2, cV, wc_obj, par, B, V, lep)
    return obs_q2int(fct, wc_obj, par, B, V, lep)

def dBR_dq2_dphi_binned(q2, phimin, phimax, wc_obj, par, B, V, lep):
    if not kinem_allowed(q2, par, B, V, lep):
        return 0
    tauB = par['tau_'+B]
    J = get_angularcoeff(q2, wc_obj, par, B, V, lep)
    dG = dG_dq2_dphi(J)
    ang_min = _angle_int(phimin)
    ang_max = _angle_int(phimax)
    return BRfac(V) * tauB * sum(
                        [y * (ang_max[a] - ang_min[a]) for a, y in dG.items()])

def BR_binned_phi(phimin, phimax, wc_obj, par, B, V, lep):
    def fct(q2):
        return dBR_dq2_dphi_binned(q2, phimin, phimax, wc_obj, par, B, V, lep)
    return obs_q2int(fct, wc_obj, par, B, V, lep)


def dBR_dq2_dphi(q2, phi, wc_obj, par, B, V, lep):
    if not kinem_allowed(q2, par, B, V, lep):
        return 0
    tauB = par['tau_'+B]
    J = get_angularcoeff(q2, wc_obj, par, B, V, lep)
    dG = dG_dq2_dphi(J)
    ang = _angle_diff(phi)
    return BRfac(V) * tauB * sum(
                        [y * ang[a] for a, y in dG.items()])

def dBR_dphi(phi, wc_obj, par, B, V, lep):
    def fct(q2):
        return dBR_dq2_dphi(q2, phi, wc_obj, par, B, V, lep)
    return obs_q2int(fct, wc_obj, par, B, V, lep)

def BRfac(V):
    if V == 'rho0' or V == 'omega':
        # factor of 1/2 for neutral rho due to rho = (uubar-ddbar)/sqrt(2)
        # and also for omega = (uubar+ddbar)/sqrt(2)
        return 1/2.
    else:
        return 1

def dBRdq2_lep(q2, wc_obj, par, B, V, lep, A):
    if not kinem_allowed(q2, par, B, V, lep):
        return 0
    tauB = par['tau_'+B]
    J = get_angularcoeff(q2, wc_obj, par, B, V, lep)
    if A is  None:
        return tauB * dGdq2(J) * BRfac(V)
    elif A == 'L':
        return tauB * dGdq2_L(J) * BRfac(V)
    elif A == 'T':
        return tauB * dGdq2_T(J) * BRfac(V)


def dBRdq2(q2, wc_obj, par, B, V, lep, A):
    if lep == 'l':
        # average of e and mu!
        return (dBRdq2_lep(q2, wc_obj, par, B, V, 'e', A) + dBRdq2_lep(q2, wc_obj, par, B, V, 'mu', A))/2
    else:
        return dBRdq2_lep(q2, wc_obj, par, B, V, lep, A)

def dBRdq2_function(B, V, lep, A):
    return lambda wc_obj, par, q2: dBRdq2(q2, wc_obj, par, B, V, lep, A)

def BR_binned(q2min, q2max, wc_obj, par, B, V, lep, A):
    def integrand(q2):
        return dBRdq2(q2, wc_obj, par, B, V, lep, A)
    return flavio.math.integrate.nintegrate(integrand, q2min, q2max)

def BR_binned_function(B, V, lep, A):
    return lambda wc_obj, par, q2min, q2max: BR_binned(q2min, q2max, wc_obj, par, B, V, lep, A)

def BR_binned_tot_function(B, V, lep, A):
    def f(wc_obj, par, q2min, q2max):
        num = BR_binned(q2min, q2max, wc_obj, par, B, V, lep, A)
        if num == 0:
            return 0
        den = BR_tot(wc_obj, par, B, V, lep, A)
        return num / den
    return f

def FL_function(B, V, lep):
    return lambda wc_obj, par, q2: FL_diff(q2, wc_obj, par, B, V, lep)

def FL_binned_function(B, V, lep):
    return lambda wc_obj, par, q2min, q2max: FL_binned(q2min, q2max, wc_obj, par, B, V, lep)

def FL_tot_function(B, V, lep):
    def f(wc_obj, par):
        mB = par['m_'+B]
        mV = par['m_'+V]
        ml = par['m_'+lep]
        q2max = (mB-mV)**2
        q2min = ml**2
        return FL_binned(q2min, q2max, wc_obj, par, B, V, lep)
    return f

def FLt_tot_function(B, V, lep):
    def f(wc_obj, par):
        def fct_J(J):
            return -J['2c']
        return Itot_norm(fct_J, wc_obj, par, B, V, lep)
    return f

def AFB_tot_function(B, V, lep):
    def f(wc_obj, par):
        def fct_J(J):
            return 3 / 8 * (2 * J['6s'] + J['6c'])
        return Itot_norm(fct_J, wc_obj, par, B, V, lep)
    return f

def I3_tot_function(B, V, lep):
    def f(wc_obj, par):
        def fct_J(J):
            return J[3]
        return Itot_norm(fct_J, wc_obj, par, B, V, lep)
    return f


def BR_binned_costhl_function(B, V, lep):
    if lep == 'l':
        return lambda wc_obj, par, clmin, clmax: (
          BR_binned_costhl(clmin, clmax, wc_obj, par, B, V, 'e')
        + BR_binned_costhl(clmin, clmax, wc_obj, par, B, V, 'mu'))/2.
    return lambda wc_obj, par, clmin, clmax: BR_binned_costhl(clmin, clmax, wc_obj, par, B, V, lep)

def BR_binned_costhV_function(B, V, lep):
    if lep == 'l':
        return lambda wc_obj, par, cVmin, cVmax: (
          BR_binned_costhV(cVmin, cVmax, wc_obj, par, B, V, 'e')
        + BR_binned_costhV(cVmin, cVmax, wc_obj, par, B, V, 'mu'))/2.
    return lambda wc_obj, par, cVmin, cVmax: BR_binned_costhV(cVmin, cVmax, wc_obj, par, B, V, lep)

def BR_binned_phi_function(B, V, lep):
    if lep == 'l':
        return lambda wc_obj, par, phimin, phimax: (
          BR_binned_phi(phimin, phimax, wc_obj, par, B, V, 'e')
        + BR_binned_phi(phimin, phimax, wc_obj, par, B, V, 'mu'))/2.
    return lambda wc_obj, par, phimin, phimax: BR_binned_phi(phimin, phimax, wc_obj, par, B, V, lep)

def dBR_dcosthl_function(B, V, lep):
    if lep == 'l':
        return lambda wc_obj, par, cl: (
          dBR_dcosthl(cl, wc_obj, par, B, V, 'e')
        + dBR_dcosthl(cl, wc_obj, par, B, V, 'mu'))/2.
    return lambda wc_obj, par, cl: dBR_dcosthl(cl, wc_obj, par, B, V, lep)

def dBR_dcosthV_function(B, V, lep):
    if lep == 'l':
        return lambda wc_obj, par, cV: (
          dBR_dcosthV(cV, wc_obj, par, B, V, 'e')
        + dBR_dcosthV(cV, wc_obj, par, B, V, 'mu'))/2.
    return lambda wc_obj, par, cV: dBR_dcosthV(cV, wc_obj, par, B, V, lep)

def dBR_dphi_function(B, V, lep):
    if lep == 'l':
        return lambda wc_obj, par, phi: (
          dBR_dphi(phi, wc_obj, par, B, V, 'e')
        + dBR_dphi(phi, wc_obj, par, B, V, 'mu'))/2.
    return lambda wc_obj, par, phi: dBR_dphi(phi, wc_obj, par, B, V, lep)

def _BR_tot(wc_obj, par, B, V, lep, A):
    mB = par['m_'+B]
    mV = par['m_'+V]
    ml = par['m_'+lep]
    q2max = (mB-mV)**2
    q2min = ml**2
    return BR_binned(q2min, q2max, wc_obj, par, B, V, lep, A)

def BR_tot(wc_obj, par, B, V, lep, A):
    if lep == 'l':
        # average of e and mu!
        return (_BR_tot(wc_obj, par, B, V, 'e', A)+_BR_tot(wc_obj, par, B, V, 'mu', A))/2.
    else:
        return _BR_tot(wc_obj, par, B, V, lep, A)

def BR_tot_function(B, V, lep, A):
    return lambda wc_obj, par: BR_tot(wc_obj, par, B, V, lep, A)

def BR_binned_leptonflavour(q2min, q2max, wc_obj, par, B, V, lnum, lden, A):
    num = BR_binned(q2min, q2max, wc_obj, par, B, V, lnum, A)
    if num == 0:
        return 0
    den = BR_binned(q2min, q2max, wc_obj, par, B, V, lden, A)
    return num/den

def BR_tot_leptonflavour(wc_obj, par, B, V, lnum, lden, A):
    num = BR_tot(wc_obj, par, B, V, lnum, A)
    if num == 0:
        return 0
    den = BR_tot(wc_obj, par, B, V, lden, A)
    return num/den

def BR_tot_leptonflavour_function(B, V, lnum, lden, A):
    return lambda wc_obj, par: BR_tot_leptonflavour(wc_obj, par, B, V, lnum, lden, A)

def BR_binned_leptonflavour_function(B, V, lnum, lden, A):
    return lambda wc_obj, par, q2min, q2max: BR_binned_leptonflavour(q2min, q2max, wc_obj, par, B, V, lnum, lden, A)


# Observable and Prediction instances

_tex = {'e': 'e', 'mu': '\mu', 'tau': r'\tau', 'l': r'\ell'}

_A = {'dBR/dq2': None, 'BR': None, '<BR>': None,
      'dBR_L/dq2': 'L', 'BR_L': 'L', '<BR_L>': 'L',
      'dBR_T/dq2': 'T', 'BR_T': 'T', '<BR_T>': 'T',
     }
_func = {'dBR/dq2': dBRdq2_function, 'BR': BR_tot_function, '<BR>': BR_binned_function,
         'dBR_L/dq2': dBRdq2_function, 'BR_L': BR_tot_function, '<BR_L>': BR_binned_function,
         'dBR_T/dq2': dBRdq2_function, 'BR_T': BR_tot_function, '<BR_T>': BR_binned_function,
         '<BR>/<cl>': BR_binned_costhl_function,
         '<BR>/<cV>': BR_binned_costhV_function,
         '<BR>/<phi>': BR_binned_phi_function,
         'dBR/dcl': dBR_dcosthl_function,
         'dBR/dcV': dBR_dcosthV_function,
         'dBR/dphi': dBR_dphi_function,
         'FL': FL_function,
         '<FL>': FL_binned_function,
         'FLtot': FL_tot_function,
         'FLttot': FLt_tot_function,
         'AFBtot': AFB_tot_function,
         'I3tot': I3_tot_function,
         }
_desc = {'dBR/dq2': r'$q^2$-differential', 'BR': 'Total', '<BR>': '$q^2$-binned',
         'dBR_L/dq2': 'Differential longitudinal', 'BR_L': 'Total longitudinal', '<BR_L>': 'Binned longitudinal',
         'dBR_T/dq2': 'Differential transverse', 'BR_T': 'Total transverse', '<BR_T>': 'Binned transverse',
         '<BR>/<cl>': r'$\cos\theta_l$-binned',
         '<BR>/<cV>': r'$\cos\theta_V$-binned',
         '<BR>/<phi>': r'$\phi$-binned',
         'dBR/dcl': r'$\cos\theta_l$-differential',
         'dBR/dcV':r'$\cos\theta_V$-differential ',
         'dBR/dphi': r'$\phi$-differential',
         'FL': r'Differential longitudinal polarization fraction',
         '<FL>': r'Binned longitudinal polarization fraction',
         'FLtot': r'Total longitudinal polarization fraction',
         'FLttot': r'Total longitudinal polarization fraction',
         'AFBtot': r'Total forward-backward asymmetry',
         'I3tot': r'$q^2$-integrated angular coefficient $I_3$',
         }
_tex_br = {'dBR/dq2': r'\frac{d\text{BR}}{dq^2}', 'BR': r'\text{BR}', '<BR>': r'\langle\text{BR}\rangle',
           'dBR_L/dq2': r'\frac{d\text{BR}_L}{dq^2}', 'BR_L': r'\text{BR}_L', '<BR_L>': r'\langle\text{BR}_L\rangle',
           'dBR_T/dq2': r'\frac{d\text{BR}_T}{dq^2}', 'BR_T': r'\text{BR}_T', '<BR_T>': r'\langle\text{BR}_T\rangle',
           '<BR>/<cl>': r'\langle\text{BR}\rangle/\Delta\cos\theta_l',
           '<BR>/<cV>': r'\langle\text{BR}\rangle/\Delta\cos\theta_V',
           '<BR>/<phi>': r'\langle\text{BR}\rangle/\Delta\phi',
           'dBR/dcl': r'\frac{d\text{BR}}{d\cos\theta_l}',
           'dBR/dcV': r'\frac{d\text{BR}}{d\cos\theta_V}',
           'dBR/dphi': r'\frac{d\text{BR}}{d\phi}',
           'FL': r'F_L',
           '<FL>': r'\langle F_L\rangle',
           'FLtot': r'F_L',
           'FLttot': r'\widetilde{F}_L',
           'AFBtot': r'A_\text{FB}',
           'I3tot': r'I_3',
            }
_args = {'dBR/dq2': ['q2'], 'BR': None, '<BR>': ['q2min', 'q2max'],
         'dBR_L/dq2': ['q2'], 'BR_L': None, '<BR_L>': ['q2min', 'q2max'],
         'dBR_T/dq2': ['q2'], 'BR_T': None, '<BR_T>': ['q2min', 'q2max'],
         '<BR>/<cl>': ['clmin', 'clmax'],
         '<BR>/<cV>': ['cVmin', 'cVmax'],
         '<BR>/<phi>': ['phimin', 'phimax'],
         'dBR/dcl': ['cl'],
         'dBR/dcV': ['cV'],
         'dBR/dphi': ['phi'],
         'FL': ['q2'],
         '<FL>': ['q2min', 'q2max'],
         'FLtot': None,
         'FLttot': None,
         'AFBtot': None,
         'I3tot': None,
         }
_hadr = {
'B0->D*': {'tex': r"B^0\to D^{\ast -}", 'B': 'B0', 'V': 'D*+', },
'B+->D*': {'tex': r"B^+\to D^{\ast 0}", 'B': 'B+', 'V': 'D*0', },
'B0->rho': {'tex': r"B^0\to \rho^-", 'B': 'B0', 'V': 'rho+', },
'B+->rho': {'tex': r"B^+\to \rho^0", 'B': 'B+', 'V': 'rho0', },
'B+->omega': {'tex': r"B^+\to \omega ", 'B': 'B+', 'V': 'omega', },
'Bs->K*': {'tex': r"B_s\to K^{* -} ", 'B': 'Bs', 'V': 'K*+', },
}
# for LF ratios we don't distinguish B+ and B0 (but take B0 because we have to choose sth)
_hadr_l = {
'B->D*': {'tex': r"B\to D^{\ast}", 'B': 'B0', 'V': 'D*+', 'decays': ['B0->D*', 'B+->D*'],},
'B->rho': {'tex': r"B\to \rho", 'B': 'B0', 'V': 'rho+', 'decays': ['B0->rho', 'B+->rho'],},
'B+->omega': {'tex': r"B^+\to \omega ", 'B': 'B+', 'V': 'omega', 'decays': ['B+->omega'],},
'Bs->K*': {'tex': r"B_s\to K^{* -} ", 'B': 'Bs', 'V': 'K*+', 'decays': ['Bs->K*'],},
}

_process_taxonomy = r'Process :: $b$ hadron decays :: Semi-leptonic tree-level decays :: $B\to V\ell\nu$ :: $'

for l in ['e', 'mu', 'tau', 'l']:
    for br in ['dBR/dq2', 'BR', '<BR>',
               'dBR_L/dq2', 'BR_L', '<BR_L>',
               'dBR_T/dq2', 'BR_T', '<BR_T>',
               '<BR>/<cl>', '<BR>/<cV>', '<BR>/<phi>',
               'dBR/dcl', 'dBR/dcV', 'dBR/dphi',
               '<FL>', 'FL', 'FLtot', 'FLttot', 'AFBtot', 'I3tot']:
        for M in _hadr.keys():
            _process_tex = _hadr[M]['tex']+_tex[l]+r"^+\nu_"+_tex[l]
            _obs_name = br + "("+M+l+"nu)"
            _obs = Observable(_obs_name)
            _obs.set_description(_desc[br] + r" branching ratio of $" + _process_tex + "$")
            _obs.tex = r'$' + _tex_br[br] + r"(" +_process_tex + ")$"
            _obs.arguments = _args[br]
            _obs.add_taxonomy(_process_taxonomy + _process_tex +  r'$')
            if br in _A:
                # for dBR/dq2, need to distinguish between total, L, and T
                Prediction(_obs_name, _func[br](_hadr[M]['B'], _hadr[M]['V'], l, A=_A[br]))
            else:
                # for other observables not
                Prediction(_obs_name, _func[br](_hadr[M]['B'], _hadr[M]['V'], l))


# Lepton flavour ratios
for l in [('mu','e'), ('tau','mu'), ('tau', 'l')]:
    for M in _hadr_l.keys():

        # binned ratio of BRs
        _obs_name = "<R"+l[0]+l[1]+">("+M+"lnu)"
        _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
        _obs.set_description(r"Ratio of partial branching ratios of $" + _hadr_l[M]['tex'] +_tex[l[0]]+r"^+ \nu_"+_tex[l[0]]+r"$" + " and " + r"$" + _hadr_l[M]['tex'] +_tex[l[1]]+r"^+ \nu_"+_tex[l[1]]+r"$")
        _obs.tex = r"$\langle R_{" + _tex[l[0]] + ' ' + _tex[l[1]] + r"} \rangle(" + _hadr_l[M]['tex'] + r"\ell^+\nu)$"
        for li in l:
            for N in _hadr_l[M]['decays']:
                # add taxonomy for both processes (e.g. B->Venu and B->Vmunu) and for charged and neutral
                _obs.add_taxonomy(_process_taxonomy + _hadr[N]['tex'] + _tex[li]+r"^+\nu_"+_tex[li]+r"$")
        Prediction(_obs_name, BR_binned_leptonflavour_function(_hadr_l[M]['B'], _hadr_l[M]['V'], l[0], l[1], A=None))

        # ratio of total BRs
        _obs_name = "R"+l[0]+l[1]+"("+M+"lnu)"
        _obs = Observable(name=_obs_name)
        _obs.set_description(r"Ratio of total branching ratios of $" + _hadr_l[M]['tex'] +_tex[l[0]]+r"^+ \nu_"+_tex[l[0]]+r"$" + " and " + r"$" + _hadr_l[M]['tex'] +_tex[l[1]]+r"^+ \nu_"+_tex[l[1]]+r"$")
        _obs.tex = r"$R_{" + _tex[l[0]] + ' ' + _tex[l[1]] + r"}(" + _hadr_l[M]['tex'] + r"\ell^+\nu)$"
        for li in l:
            for N in _hadr_l[M]['decays']:
                # add taxonomy for both processes (e.g. B->Venu and B->Vmunu) and for charged and neutral
                _obs.add_taxonomy(_process_taxonomy + _hadr[N]['tex'] +_tex[li]+r"^+\nu_"+_tex[li]+r"$")
        Prediction(_obs_name, BR_tot_leptonflavour_function(_hadr_l[M]['B'], _hadr_l[M]['V'], l[0], l[1], A=None))


# B->D*taunu normalized binned BR
_obs_name = "<BR>/BR(B->D*taunu)"
_obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
_obs.set_description(r"Relative partial branching ratio of $B\to D^\ast\tau^+\nu$")
_obs.tex = r"$\frac{\langle \text{BR} \rangle}{\text{BR}}(B\to D^\ast\tau^+\nu)$"
for M in ['B+->D*', 'B0->D*']:
    _process_tex = _hadr[M]['tex'] + r"\tau^+\nu"
    _obs.add_taxonomy(_process_taxonomy + _process_tex + r"$")
Prediction(_obs_name, BR_binned_tot_function('B0', 'D*+', 'tau', A=None))
