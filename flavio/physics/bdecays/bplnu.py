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
    return AuxiliaryQuantity[ff_name].prediction(par_dict=par, wc_obj=None, q2=q2)

def get_angularcoeff(q2, wc_obj, par, B, P, lep):
    Jlist = [_get_angularcoeff(q2, wc_obj, par, B, P, lep, nu)
             for nu in ['e', 'mu', 'tau']]
    J = {}
    J['a'] = sum([JJ['a'] for JJ in Jlist])
    J['b'] = sum([JJ['b'] for JJ in Jlist])
    J['c'] = sum([JJ['c'] for JJ in Jlist])
    return J


def _get_angularcoeff(q2, wc_obj, par, B, P, lep, nu):
    scale = config['renormalization scale']['bpll']
    mb = running.get_mb(par, scale)
    wc = get_wceff_fccc(wc_obj, par, meson_quark[(B,P)], lep, nu, mb, scale, nf=5)
    if lep != nu and all(C == 0 for C in wc.values()):
        return {'a': 0, 'b': 0, 'c': 0}  # if all WCs vanish, so does the AC!
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mP = par['m_'+P]
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

def BR_binned_leptonflavour(q2min, q2max, wc_obj, par, B, P, lnum, lden):
    num = BR_binned(q2min, q2max, wc_obj, par, B, P, lnum)
    if num == 0:
        return 0
    den = BR_binned(q2min, q2max, wc_obj, par, B, P, lden)
    return num/den

def BR_binned_leptonflavour_function(B, P, lnum, lden):
    return lambda wc_obj, par, q2min, q2max: BR_binned_leptonflavour(q2min, q2max, wc_obj, par, B, P, lnum, lden)


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

def BR_binned_tot_function(B, P, lep):
    def f(wc_obj, par, q2min, q2max):
        num = BR_binned(q2min, q2max, wc_obj, par, B, P, lep)
        if num == 0:
            return 0
        den = BR_tot(wc_obj, par, B, P, lep)
        return num / den
    return f

def BR_tot_leptonflavour(wc_obj, par, B, P, lnum, lden):
    num = BR_tot(wc_obj, par, B, P, lnum)
    if num == 0:
        return 0
    den = BR_tot(wc_obj, par, B, P, lden)
    return num/den

def BR_tot_leptonflavour_function(B, P, lnum, lden):
    return lambda wc_obj, par: BR_tot_leptonflavour(wc_obj, par, B, P, lnum, lden)

# Observable and Prediction instances

_tex = {'e': 'e', 'mu': '\mu', 'tau': r'\tau', 'l': r'\ell'}
_func = {'dBR/dq2': dBRdq2_function, 'BR': BR_tot_function, '<BR>': BR_binned_function}
_desc = {'dBR/dq2': 'Differential', 'BR': 'Total', '<BR>': 'Binned'}
_tex_br = {'dBR/dq2': r'\frac{d\text{BR}}{dq^2}', 'BR': r'\text{BR}', '<BR>': r'\langle\text{BR}\rangle'}
_args = {'dBR/dq2': ['q2'], 'BR': None, '<BR>': ['q2min', 'q2max']}

_hadr = {
'B+->D': {'tex': r"B^+\to D^0", 'B': 'B+', 'P': 'D0', },
'B0->D': {'tex': r"B^0\to D^- ", 'B': 'B0', 'P': 'D+', },
'B+->pi': {'tex': r"B^+\to \pi^0", 'B': 'B+', 'P': 'pi0', },
'B0->pi': {'tex': r"B^0\to \pi^- ", 'B': 'B0', 'P': 'pi+', },
}

# for LF ratios we don't distinguish B+ and B0 (but take B0 because we have to choose sth)
_hadr_l = {
'B->D': {'tex': r"B\to D", 'B': 'B0', 'P': 'D+', 'decays': ['B+->D', 'B0->D'],},
'B->pi': {'tex': r"B\to \pi ", 'B': 'B0', 'P': 'pi+', 'decays': ['B+->pi', 'B0->pi'],},
}


_process_taxonomy = r'Process :: $b$ hadron decays :: Semi-leptonic tree-level decays :: $B\to P\ell\nu$ :: $'

for l in ['e', 'mu', 'tau', 'l']:
    for M in _hadr.keys():
        for br in ['dBR/dq2', 'BR', '<BR>']:
            _obs_name = br + "("+M+l+"nu)"
            _process_tex = _hadr[M]['tex']+_tex[l]+r"^+\nu_"+_tex[l]
            _obs = Observable(_obs_name)
            _obs.set_description(_desc[br] + r" branching ratio of $" + _process_tex + r"$")
            _obs.tex = r'$' + _tex_br[br] + r"(" + _process_tex + r")$"
            _obs.arguments = _args[br]
            _obs.add_taxonomy(_process_taxonomy + _process_tex + r"$")
            Prediction(_obs_name, _func[br](_hadr[M]['B'], _hadr[M]['P'], l))

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
                # add taxonomy for both processes (e.g. B->Penu and B->Pmunu) and for charged and neutral
                _obs.add_taxonomy(_process_taxonomy + _hadr[N]['tex'] + _tex[li]+r"^+\nu_"+_tex[li]+r"$")
        Prediction(_obs_name, BR_binned_leptonflavour_function(_hadr_l[M]['B'], _hadr_l[M]['P'], l[0], l[1]))

        # ratio of total BRs
        _obs_name = "R"+l[0]+l[1]+"("+M+"lnu)"
        _obs = Observable(name=_obs_name)
        _obs.set_description(r"Ratio of total branching ratios of $" + _hadr_l[M]['tex'] +_tex[l[0]]+r"^+ \nu_"+_tex[l[0]]+r"$" + " and " + r"$" + _hadr_l[M]['tex'] +_tex[l[1]]+r"^+ \nu_"+_tex[l[1]]+r"$")
        _obs.tex = r"$R_{" + _tex[l[0]] + ' ' + _tex[l[1]] + r"}(" + _hadr_l[M]['tex'] + r"\ell^+\nu)$"
        for li in l:
            for N in _hadr_l[M]['decays']:
                # add taxonomy for both processes (e.g. B->Penu and B->Pmunu) and for charged and neutral
                _obs.add_taxonomy(_process_taxonomy + _hadr[N]['tex'] +_tex[li]+r"^+\nu_"+_tex[li]+r"$")
        Prediction(_obs_name, BR_tot_leptonflavour_function(_hadr_l[M]['B'], _hadr_l[M]['P'], l[0], l[1]))


# B->Dtaunu normalized binned BR
_obs_name = "<BR>/BR(B->Dtaunu)"
_obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
_obs.set_description(r"Relative partial branching ratio of $B\to D\tau^+\nu$")
_obs.tex = r"$\frac{\langle \text{BR} \rangle}{\text{BR}}(B\to D\tau^+\nu)$"
for M in ['B+->D', 'B0->D']:
    _process_tex = _hadr[M]['tex'] + r"\tau^+\nu"
    _obs.add_taxonomy(_process_taxonomy + _process_tex + r"$")
Prediction(_obs_name, BR_binned_tot_function('B0', 'D+', 'tau'))
