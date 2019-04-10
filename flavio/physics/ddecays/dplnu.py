r"""Functions for exclusive $D\to P\ell\nu$ decays.

Copied from `flavio.physics.bdecays.bplnu`
"""

from math import sqrt
import flavio
from flavio.physics import ckm
from flavio.classes import AuxiliaryQuantity
from flavio.config import config
from flavio.physics.running import running
from flavio.physics.bdecays import angular
from flavio.physics.bdecays.wilsoncoefficients import get_wceff_fccc
from flavio.classes import Observable, Prediction


meson_quark = {
    ('D+', 'pi0'): 'cd',
    ('D0', 'pi+'): 'cd',
    ('D+', 'K0'): 'cs',
    ('D0', 'K+'): 'cs',
}

meson_ff = {
    ('D+', 'pi0'): 'D->pi',
    ('D0', 'pi+'): 'D->pi',
    ('D+', 'K0'): 'D->K',
    ('D0', 'K+'): 'D->K',
}

def prefactor(q2, par, D, P, lep):
    GF = par['GF']
    ml = par['m_'+lep]
    qi_qj = meson_quark[(D, P)]
    if qi_qj == 'cd':
        Vij = ckm.get_ckm(par)[1, 0] # V_{cd} for c->d transitions
    if qi_qj == 'cs':
        Vij = ckm.get_ckm(par)[1, 1] # V_{cs} for c->s transitions
    if q2 <= ml**2:
        return 0
    return 4 * GF / sqrt(2) * Vij


def get_ff(q2, par, D, P):
    ff_name = meson_ff[(D, P)] + ' form factor'
    return AuxiliaryQuantity[ff_name].prediction(par_dict=par, wc_obj=None, q2=q2)


def get_angularcoeff(q2, wc_obj, par, D, P, lep):
    Jlist = [_get_angularcoeff(q2, wc_obj, par, D, P, lep, nu)
             for nu in ['e', 'mu', 'tau']]
    J = {}
    J['a'] = sum([JJ['a'] for JJ in Jlist])
    J['b'] = sum([JJ['b'] for JJ in Jlist])
    J['c'] = sum([JJ['c'] for JJ in Jlist])
    return J


def _get_angularcoeff(q2, wc_obj, par, D, P, lep, nu):
    scale = config['renormalization scale']['dpll']
    mc = running.get_mc(par, scale)
    wc = get_wceff_fccc(wc_obj, par, meson_quark[(D, P)][::-1], lep, nu, None, scale, nf=4)
    if lep != nu and all(C == 0 for C in wc.values()):
        return {'a': 0, 'b': 0, 'c': 0}  # if all WCs vanish, so does the AC!
    ml = par['m_' + lep]
    mD = par['m_' + D]
    mP = par['m_' + P]
    N = prefactor(q2, par, D, P, lep)
    ff = get_ff(q2, par, D, P)
    qi_qj = meson_quark[(D, P)]
    if qi_qj == 'cd':
        mlight = running.get_md(par, scale)
    if qi_qj == 'cs':
        mlight = running.get_ms(par, scale)
    h = angular.helicity_amps_p(q2, mD, mP, mc, mlight, ml, 0, ff, wc, N)
    J = angular.angularcoeffs_general_p(h, q2, mD, mP, mc, mlight, ml, 0)
    return J


def dGdq2(J):
    return 2 * (J['a'] + J['c']/3.)


def dBRdq2_lep(q2, wc_obj, par, D, P, lep):
    ml = par['m_' + lep]
    mD = par['m_' + D]
    mP = par['m_' + P]
    if q2 < ml**2 or q2 > (mD-mP)**2:
        return 0
    tauD = par['tau_' + D]
    J = get_angularcoeff(q2, wc_obj, par, D, P, lep)
    if P == 'pi0':
        # factor of 1/2 for neutral pi due to pi = (uubar-ddbar)/sqrt(2)
        return tauD * dGdq2(J) / 2.
    return tauD * dGdq2(J)


def dBRdq2(q2, wc_obj, par, D, P, lep):
    if lep == 'l':
        # average of e and mu!
        return (dBRdq2_lep(q2, wc_obj, par, D, P, 'e') + dBRdq2_lep(q2, wc_obj, par, D, P, 'mu'))/2
    else:
        return dBRdq2_lep(q2, wc_obj, par, D, P, lep)


def dBRdq2_function(D, P, lep):
    return lambda wc_obj, par, q2: dBRdq2(q2, wc_obj, par, D, P, lep)


def BR_binned(q2min, q2max, wc_obj, par, D, P, lep):
    def integrand(q2):
        return dBRdq2(q2, wc_obj, par, D, P, lep)
    return flavio.math.integrate.nintegrate(integrand, q2min, q2max)


def BR_binned_function(D, P, lep):
    return lambda wc_obj, par, q2min, q2max: BR_binned(q2min, q2max, wc_obj, par, D, P, lep)


def BR_binned_leptonflavour(q2min, q2max, wc_obj, par, D, P, lnum, lden):
    num = BR_binned(q2min, q2max, wc_obj, par, D, P, lnum)
    if num == 0:
        return 0
    den = BR_binned(q2min, q2max, wc_obj, par, D, P, lden)
    return num / den


def BR_binned_leptonflavour_function(D, P, lnum, lden):
    return lambda wc_obj, par, q2min, q2max: BR_binned_leptonflavour(q2min, q2max, wc_obj, par, D, P, lnum, lden)


def _BR_tot(wc_obj, par, D, P, lep):
    mD = par['m_'+D]
    mP = par['m_'+P]
    ml = par['m_'+lep]
    q2max = (mD-mP)**2
    q2min = ml**2
    return BR_binned(q2min, q2max, wc_obj, par, D, P, lep)


def BR_tot(wc_obj, par, D, P, lep):
    if lep == 'l':
        # average of e and mu!
        return (_BR_tot(wc_obj, par, D, P, 'e')+_BR_tot(wc_obj, par, D, P, 'mu'))/2.
    else:
        return _BR_tot(wc_obj, par, D, P, lep)


def BR_tot_function(D, P, lep):
    return lambda wc_obj, par: BR_tot(wc_obj, par, D, P, lep)


def BR_binned_tot_function(D, P, lep):
    def f(wc_obj, par, q2min, q2max):
        num = BR_binned(q2min, q2max, wc_obj, par, D, P, lep)
        if num == 0:
            return 0
        den = BR_tot(wc_obj, par, D, P, lep)
        return num / den
    return f


def BR_tot_leptonflavour(wc_obj, par, D, P, lnum, lden):
    num = BR_tot(wc_obj, par, D, P, lnum)
    if num == 0:
        return 0
    den = BR_tot(wc_obj, par, D, P, lden)
    return num/den


def BR_tot_leptonflavour_function(D, P, lnum, lden):
    return lambda wc_obj, par: BR_tot_leptonflavour(wc_obj, par, D, P, lnum, lden)


# Observable and Prediction instances

_tex = {'e': 'e', 'mu': '\mu', 'l': r'\ell'}
_func = {'dBR/dq2': dBRdq2_function, 'BR': BR_tot_function, '<BR>': BR_binned_function}
_desc = {'dBR/dq2': 'Differential', 'BR': 'Total', '<BR>': 'Binned'}
_tex_br = {'dBR/dq2': r'\frac{d\text{BR}}{dq^2}', 'BR': r'\text{BR}', '<BR>': r'\langle\text{BR}\rangle'}
_args = {'dBR/dq2': ['q2'], 'BR': None, '<BR>': ['q2min', 'q2max']}

_hadr = {
'D+->K': {'tex': r"D^+\to K^0", 'D': 'D+', 'P': 'K0', },
'D0->K': {'tex': r"D^0\to K^- ", 'D': 'D0', 'P': 'K+', },
'D+->pi': {'tex': r"D^+\to \pi^0", 'D': 'D+', 'P': 'pi0', },
'D0->pi': {'tex': r"D^0\to \pi^- ", 'D': 'D0', 'P': 'pi+', },
}

# for LF ratios we don't distinguish D+ and D0 (but take D0 because we have to choose sth)
_hadr_l = {
'D->K': {'tex': r"D\to K", 'D': 'D0', 'P': 'K+', 'decays': ['D+->K', 'D0->K'],},
'D->pi': {'tex': r"D\to \pi ", 'D': 'D0', 'P': 'pi+', 'decays': ['D+->pi', 'D0->pi'],},
}


_process_taxonomy = r'Process :: $c$ hadron decays :: Semi-leptonic tree-level decays :: $D\to P\ell\nu$ :: $'

for l in ['e', 'mu', 'l']:
    for M in _hadr.keys():
        for br in ['dBR/dq2', 'BR', '<BR>']:
            _obs_name = br + "("+M+l+"nu)"
            _process_tex = _hadr[M]['tex']+_tex[l]+r"^+\nu_"+_tex[l]
            _obs = Observable(_obs_name)
            _obs.set_description(_desc[br] + r" branching ratio of $" + _process_tex + r"$")
            _obs.tex = r'$' + _tex_br[br] + r"(" + _process_tex + r")$"
            _obs.arguments = _args[br]
            _obs.add_taxonomy(_process_taxonomy + _process_tex + r"$")
            Prediction(_obs_name, _func[br](_hadr[M]['D'], _hadr[M]['P'], l))

# Lepton flavour ratios
for l in [('mu','e')]:
    for M in _hadr_l.keys():

        # binned ratio of BRs
        _obs_name = "<R"+l[0]+l[1]+">("+M+"lnu)"
        _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
        _obs.set_description(r"Ratio of partial branching ratios of $" + _hadr_l[M]['tex'] +_tex[l[0]]+r"^+ \nu_"+_tex[l[0]]+r"$" + " and " + r"$" + _hadr_l[M]['tex'] +_tex[l[1]]+r"^+ \nu_"+_tex[l[1]]+r"$")
        _obs.tex = r"$\langle R_{" + _tex[l[0]] + ' ' + _tex[l[1]] + r"} \rangle(" + _hadr_l[M]['tex'] + r"\ell^+\nu)$"
        for li in l:
            for N in _hadr_l[M]['decays']:
                # add taxonomy for both processes (e.g. D->Penu and D->Pmunu) and for charged and neutral
                _obs.add_taxonomy(_process_taxonomy + _hadr[N]['tex'] + _tex[li]+r"^+\nu_"+_tex[li]+r"$")
        Prediction(_obs_name, BR_binned_leptonflavour_function(_hadr_l[M]['D'], _hadr_l[M]['P'], l[0], l[1]))

        # ratio of total BRs
        _obs_name = "R"+l[0]+l[1]+"("+M+"lnu)"
        _obs = Observable(name=_obs_name)
        _obs.set_description(r"Ratio of total branching ratios of $" + _hadr_l[M]['tex'] +_tex[l[0]]+r"^+ \nu_"+_tex[l[0]]+r"$" + " and " + r"$" + _hadr_l[M]['tex'] +_tex[l[1]]+r"^+ \nu_"+_tex[l[1]]+r"$")
        _obs.tex = r"$R_{" + _tex[l[0]] + ' ' + _tex[l[1]] + r"}(" + _hadr_l[M]['tex'] + r"\ell^+\nu)$"
        for li in l:
            for N in _hadr_l[M]['decays']:
                # add taxonomy for both processes (e.g. D->Penu and D->Pmunu) and for charged and neutral
                _obs.add_taxonomy(_process_taxonomy + _hadr[N]['tex'] +_tex[li]+r"^+\nu_"+_tex[li]+r"$")
        Prediction(_obs_name, BR_tot_leptonflavour_function(_hadr_l[M]['D'], _hadr_l[M]['P'], l[0], l[1]))
