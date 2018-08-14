r"""Functions for $K\to \pi\ell\nu$ decays."""

from math import sqrt
import flavio
from flavio.classes import Observable, Prediction


def get_ff(q2, par, K):
    ff_name = 'K->pi form factor'
    ff_K0 = flavio.classes.AuxiliaryQuantity[ff_name].prediction(par_dict=par, wc_obj=None, q2=q2)
    if K == 'KL':
        return ff_K0
    elif K == 'K+':
        # isospin breaking correction for K+->pi0lnu: multiply all FFs by 1+delta
        return {k: (par['K->pi delta_K+pi0'] + 1)*v for k,v in ff_K0.items()}

def get_angularcoeff(q2, wc_obj, par, K, P, lep):
    Jlist = [_get_angularcoeff(q2, wc_obj, par, K, P, lep, nu)
             for nu in ['e', 'mu', 'tau']]
    J = {}
    J['a'] = sum([JJ['a'] for JJ in Jlist])
    J['b'] = sum([JJ['b'] for JJ in Jlist])
    J['c'] = sum([JJ['c'] for JJ in Jlist])
    return J

def _get_angularcoeff(q2, wc_obj, par, K, P, lep, nu):
    GF = par['GF']
    ml = par['m_'+lep]
    mK = par['m_'+K]
    mP = par['m_'+P]
    Vus = flavio.physics.ckm.get_ckm(par)[0,1]
    # renormalization scale is m_rho
    scale = par['m_rho0']
    ms = flavio.physics.running.running.get_ms(par, scale)
    wc = flavio.physics.bdecays.wilsoncoefficients.get_wceff_fccc(wc_obj, par, 'su', lep, nu, ms, scale, nf=3)
    N = 4*GF/sqrt(2)*Vus
    ff = get_ff(q2, par, K)
    h = flavio.physics.bdecays.angular.helicity_amps_p(q2, mK, mP, ms, 0, ml, 0, ff, wc, N)
    J = flavio.physics.bdecays.angular.angularcoeffs_general_p(h, q2, mK, mP, ms, 0, ml, 0)
    return J


def dGdq2(J):
    return 2 * (J['a'] + J['c']/3.)

def dBRdq2(q2, wc_obj, par, K, P, lep):
    ml = par['m_'+lep]
    mK = par['m_'+K]
    mP = par['m_'+P]
    if q2 < ml**2 or q2 > (mK-mP)**2:
        return 0
    tauK = par['tau_'+K]
    J = get_angularcoeff(q2, wc_obj, par, K, P, lep)
    if P == 'pi0':
        # factor of 1/2 for neutral pi due to pi = (uubar-ddbar)/sqrt(2)
        return tauK * dGdq2(J) / 2.
    if K == 'K+':
        deltaEM = par['K+' + lep + '3 delta_EM'] # e.g. 'K+e3 delta_EM'
    elif K == 'KL':
        deltaEM = par['K0' + lep + '3 delta_EM'] # e.g. 'K+e3 delta_EM'
    return tauK * dGdq2(J) * (1 + deltaEM)**2

def BR_binned(q2min, q2max, wc_obj, par, K, P, lep):
    def integrand(q2):
        return dBRdq2(q2, wc_obj, par, K, P, lep)
    return flavio.math.integrate.nintegrate(integrand, q2min, q2max)

def BR_tot(wc_obj, par, K, P, lep):
    mK = par['m_'+K]
    mP = par['m_'+P]
    ml = par['m_'+lep]
    q2max = (mK-mP)**2
    q2min = ml**2
    return BR_binned(q2min, q2max, wc_obj, par, K, P, lep)

def BR_tot_function(K, P, lep):
    return lambda wc_obj, par: BR_tot(wc_obj, par, K, P, lep)


# Observable and Prediction instances

_tex = {'e': 'e', 'mu': '\mu', 'l': r'\ell'}
_tex_br = {'dBR/dq2': r'\frac{d\text{BR}}{dq^2}', 'BR': r'\text{BR}', '<BR>': r'\langle\text{BR}\rangle'}
_args = {'dBR/dq2': ['q2'], 'BR': None, '<BR>': ['q2min', 'q2max']}
_hadr = {
'KL->pi': {'tex': r"K_L\to \pi^+", 'K': 'KL', 'P': 'pi+', },
'K+->pi': {'tex': r"K^+\to \pi^0", 'K': 'K+', 'P': 'pi0', },
}

for l in ['e', 'mu', 'l']:
    for M in _hadr.keys():
        _process_tex = _hadr[M]['tex']+_tex[l]+r"^+\nu_"+_tex[l]
        _process_taxonomy = r'Process :: $s$ hadron decays :: Semi-leptonic tree-level decays :: $K\to P\ell\nu$ :: $' + _process_tex + r"$"

        _obs_name = "BR("+M+l+"nu)"
        _obs = Observable(_obs_name)
        _obs.set_description(r"Total branching ratio of $" + _process_tex + r"$")
        _obs.tex = r"$\text{BR}(" + _process_tex + r")$"
        _obs.add_taxonomy(_process_taxonomy)
        Prediction(_obs_name, BR_tot_function(_hadr[M]['K'], _hadr[M]['P'], l))
