r"""Functions for exclusive $\Lambda_b\to \Lambda\ell^+\ell^-$ decays."""

import flavio
from math import sqrt,pi
from flavio.physics.bdecays.common import lambda_K, beta_l, meson_quark, meson_ff
from flavio.classes import Observable, Prediction, AuxiliaryQuantity
from flavio.physics.common import conjugate_par, conjugate_wc, add_dict
import warnings


def helicity_amps(q2, mLb, mL, ff):
    r"""$\Lambda_b\to \Lambda\ell^+\ell^-$ helicity amplitudes.

    See (3.12)-(3.15) of arXiv:1410.2115."""
    sp = (mLb + mL)**2 - q2
    sm = (mLb - mL)**2 - q2
    H = {}
    H['0V++'] = ff['fV0'] * (mLb + mL)/sqrt(q2) * sqrt(sm)
    H['+V-+'] = -ff['fVperp'] * sqrt(2*sm)
    H['0A++'] = ff['fA0'] * (mLb - mL)/sqrt(q2) * sqrt(sp)
    H['+A-+'] = -ff['fAperp'] * sqrt(2*sp)
    H['0T++'] = -ff['fT0'] * sqrt(q2) * sqrt(sm)
    H['+T-+'] = ff['fTperp'] * (mLb + mL) * sqrt(2*sm)
    H['0T5++'] = ff['fT50'] * sqrt(q2) * sqrt(sp)
    H['+T5-+'] = -ff['fT5perp'] * (mLb - mL) * sqrt(2*sp)
    H['0V--'] = H['0V++']
    H['+V+-'] = H['+V-+']
    H['0A--'] = -H['0A++']
    H['+A+-'] = -H['+A-+']
    H['0T--'] = H['0T++']
    H['+T+-'] = H['+T-+']
    H['0T5--'] = -H['0T5++']
    H['+T5+-'] = -H['+T5-+']
    return H

def transverity_amps(ha, q2, mLb, mL, mqh, mql, wc, prefactor):
    r"""Transversity amplitudes for $\Lambda_b\to \Lambda\ell^+\ell^-$.

    See (3.16) of arXiv:1410.2115."""
    C910Lpl = (wc['v'] - wc['a']) + (wc['vp'] - wc['ap'])
    C910Rpl = (wc['v'] + wc['a']) + (wc['vp'] + wc['ap'])
    C910Lmi = (wc['v'] - wc['a']) - (wc['vp'] - wc['ap'])
    C910Rmi = (wc['v'] + wc['a']) - (wc['vp'] + wc['ap'])
    A = {}
    A['perp1', 'L'] = +sqrt(2)*( C910Lpl *ha['+V-+']
                                 - 2*mqh*(wc['7']+wc['7p'])/q2 * ha['+T-+'] )
    A['perp1', 'R'] = +sqrt(2)*( C910Rpl *ha['+V-+']
                                 - 2*mqh*(wc['7']+wc['7p'])/q2 * ha['+T-+'] )
    A['para1', 'L'] = -sqrt(2)*( C910Lmi *ha['+A-+']
                                 + 2*mqh*(wc['7']-wc['7p'])/q2 * ha['+T5-+'] )
    A['para1', 'R'] = -sqrt(2)*( C910Rmi *ha['+A-+']
                                 + 2*mqh*(wc['7']-wc['7p'])/q2 * ha['+T5-+'] )
    A['perp0', 'L'] = +sqrt(2)*( C910Lpl *ha['0V++']
                                 - 2*mqh*(wc['7']+wc['7p'])/q2 * ha['0T++'] )
    A['perp0', 'R'] = +sqrt(2)*( C910Rpl *ha['0V++']
                                 - 2*mqh*(wc['7']+wc['7p'])/q2 * ha['0T++'] )
    A['para0', 'L'] = -sqrt(2)*( C910Lmi *ha['0A++']
                                 + 2*mqh*(wc['7']-wc['7p'])/q2 * ha['0T5++'] )
    A['para0', 'R'] = -sqrt(2)*( C910Rmi *ha['0A++']
                                 + 2*mqh*(wc['7']-wc['7p'])/q2 * ha['0T5++'] )
    return {k: prefactor*v for k, v in A.items()}

def angular_coefficients(ta, alpha, Plb):
    r"""Angular coefficients of $\Lambda_b\to \Lambda\ell^+\ell^-$ in terms of
    transversity amplitudes and decay parameter $\alpha$.

    See (3.29)-(3.32) of arXiv:1410.2115 for the unpolarised case,
    and (15)-(17) of arXiv:1710.00746 for the polarised one."""
    flavio.citations.register("Boer:2014kda")
    flavio.citations.register("Blake:2017une")

    K = {}
    K['1ss'] = 1/4.*(   abs(ta['perp1', 'R'])**2 + abs(ta['perp1', 'L'])**2
                      + abs(ta['para1', 'R'])**2 + abs(ta['para1', 'L'])**2
                      + 2*abs(ta['perp0', 'R'])**2 + 2*abs(ta['perp0', 'L'])**2
                      + 2*abs(ta['para0', 'R'])**2 + 2*abs(ta['para0', 'L'])**2 )
    K['1cc'] = 1/2.*(   abs(ta['perp1', 'R'])**2 + abs(ta['perp1', 'L'])**2
                      + abs(ta['para1', 'R'])**2 + abs(ta['para1', 'L'])**2 )
    K['1c'] = -(  ta['perp1', 'R'] * ta['para1', 'R'].conj()
                - ta['perp1', 'L'] * ta['para1', 'L'].conj() ).real
    K['2ss'] = alpha/2. * (  ta['perp1', 'R'] * ta['para1', 'R'].conj()
                       + 2 * ta['perp0', 'R'] * ta['para0', 'R'].conj()
                           + ta['perp1', 'L'] * ta['para1', 'L'].conj()
                       + 2 * ta['perp0', 'L'] * ta['para0', 'L'].conj() ).real
    K['2cc'] = alpha * (  ta['perp1', 'R'] * ta['para1', 'R'].conj()
                        + ta['perp1', 'L'] * ta['para1', 'L'].conj() ).real
    K['2c'] = -alpha/2.*(   abs(ta['perp1', 'R'])**2 - abs(ta['perp1', 'L'])**2
                          + abs(ta['para1', 'R'])**2 - abs(ta['para1', 'L'])**2 )
    K['3sc'] = alpha/sqrt(2) * ( ta['perp1', 'R'] * ta['perp0', 'R'].conj()
                               - ta['para1', 'R'] * ta['para0', 'R'].conj()
                               + ta['perp1', 'L'] * ta['perp0', 'L'].conj()
                               - ta['para1', 'L'] * ta['para0', 'L'].conj() ).imag
    K['3s'] = alpha/sqrt(2) * ( ta['perp1', 'R'] * ta['para0', 'R'].conj()
                              - ta['para1', 'R'] * ta['perp0', 'R'].conj()
                              - ta['perp1', 'L'] * ta['para0', 'L'].conj()
                              + ta['para1', 'L'] * ta['perp0', 'L'].conj() ).imag
    K['4sc'] = alpha/sqrt(2) * ( ta['perp1', 'R'] * ta['para0', 'R'].conj()
                               - ta['para1', 'R'] * ta['perp0', 'R'].conj()
                               + ta['perp1', 'L'] * ta['para0', 'L'].conj()
                               - ta['para1', 'L'] * ta['perp0', 'L'].conj() ).imag
    K['4s'] = alpha/sqrt(2) * ( ta['perp1', 'R'] * ta['perp0', 'R'].conj()
                              - ta['para1', 'R'] * ta['para0', 'R'].conj()
                              - ta['perp1', 'L'] * ta['perp0', 'L'].conj()
                              + ta['para1', 'L'] * ta['para0', 'L'].conj() ).imag

    # The remaining 24 observables depend on the polarization of the Lambdab
    # Some are related to the unpolarized observables
    K['13'] = - Plb * K['2c']
    K['16'] = - Plb * K['1c']
    # In the massless lepton limit some observables are fairly simple
    K['15'] = - alpha * Plb * K['1cc']
    K['29'] = 0
    K['31'] = 0

    # The remaining ones are more complicated but often represent the real and imaginary
    # parts of the same expression, so they are defined in pairs
    K12K34 = - 0.5 * Plb * ( ta['perp1', 'R'] * ta['para1', 'R'].conj()
                           + ta['perp1', 'L'] * ta['para1', 'L'].conj() )
    K21K27 = (1/sqrt(2)) * Plb * (ta['para1', 'R'] * ta['para0', 'R'].conj()
                                + ta['para1', 'L'] * ta['para0', 'L'].conj()
                                + ta['perp1', 'R'] * ta['perp0', 'R'].conj()
                                + ta['perp1', 'L'] * ta['perp0', 'L'].conj() )
    K22K28 = -(1/sqrt(2)) * Plb * (ta['para1', 'R'] * ta['perp0', 'R'].conj()
                                + ta['perp1', 'R'] * ta['para0', 'R'].conj()
                                - ta['perp1', 'L'] * ta['para0', 'L'].conj()
                                - ta['para1', 'L'] * ta['perp0', 'L'].conj() )
    K23K25 = -(1/sqrt(2)) * Plb * (ta['para1', 'R'] * ta['perp0', 'R'].conj()
                                + ta['para1', 'L'] * ta['perp0', 'L'].conj()
                                + ta['perp1', 'R'] * ta['para0', 'R'].conj()
                                + ta['perp1', 'L'] * ta['para0', 'L'].conj() )
    K24K26 = (1/sqrt(2)) * Plb * (ta['para1', 'R'] * ta['para0', 'R'].conj()
                                - ta['para1', 'L'] * ta['para0', 'L'].conj()
                                + ta['perp1', 'R'] * ta['perp0', 'R'].conj()
                                - ta['perp1', 'L'] * ta['perp0', 'L'].conj() )

    K['11'] = - 0.5 * Plb * ( ta['para1', 'R'] * ta['perp1', 'R'].conj()
                            + ta['para1', 'L'] * ta['perp1', 'L'].conj()
                          - 2*ta['para0', 'R'] * ta['perp0', 'R'].conj()
                          - 2*ta['para0', 'L'] * ta['perp0', 'L'].conj() ).real
    K['12'] = 2 * K12K34.real # Conjugated the argument wrt. the paper but that's irrelevant for the real part
    K['14'] = - 0.25 * alpha * Plb * (  abs(ta['perp1', 'R'])**2 +   abs(ta['perp1', 'L'])**2
                                    +   abs(ta['para1', 'R'])**2 +   abs(ta['para1', 'L'])**2
                                    - 2*abs(ta['perp0', 'R'])**2 - 2*abs(ta['perp0', 'L'])**2
                                    - 2*abs(ta['para0', 'R'])**2 - 2*abs(ta['para0', 'L'])**2 )
    K['17'] = - (1/sqrt(2)) * alpha * Plb * (ta['para1', 'R'] * ta['para0', 'R'].conj() 
                                           + ta['para1', 'L'] * ta['para0', 'L'].conj()
                                           - ta['perp1', 'R'] * ta['perp0', 'R'].conj()
                                           - ta['perp1', 'L'] * ta['perp0', 'L'].conj() ).real
    K['18'] = - (1/sqrt(2)) * alpha * Plb * (ta['para1', 'R'] * ta['perp0', 'R'].conj()
                                           + ta['perp1', 'L'] * ta['para0', 'L'].conj()
                                           - ta['perp1', 'R'] * ta['para0', 'R'].conj()
                                           - ta['para1', 'L'] * ta['perp0', 'L'].conj() ).real
    K['19'] = - (1/sqrt(2)) * alpha * Plb * (ta['para1', 'R'] * ta['perp0', 'R'].conj()
                                           + ta['para1', 'L'] * ta['perp0', 'L'].conj()
                                           - ta['perp1', 'R'] * ta['para0', 'R'].conj()
                                           - ta['perp1', 'L'] * ta['para0', 'L'].conj() ).imag
    K['20'] = - (1/sqrt(2)) * alpha * Plb * (ta['para1', 'R'] * ta['para0', 'R'].conj()
                                           + ta['perp1', 'L'] * ta['perp0', 'L'].conj()
                                           - ta['perp1', 'R'] * ta['perp0', 'R'].conj()
                                           - ta['para1', 'L'] * ta['para0', 'L'].conj() ).imag
    K['21'] = K21K27.imag
    K['22'] = K22K28.imag
    K['23'] = K23K25.real
    K['24'] = K24K26.real
    K['25'] = - alpha * K23K25.imag
    K['26'] = - alpha * K24K26.imag
    K['27'] = - alpha * K21K27.real
    K['28'] = - alpha * K22K28.real
    K['30'] = 0.5 * alpha * Plb * 2 * (ta['perp0', 'R'] * ta['para0', 'R'].conj()
                                     + ta['perp0', 'L'] * ta['para0', 'L'].conj() ).imag
    K['32'] = 0.25 * alpha * Plb * 2 * (abs(ta['perp0', 'R'])**2 + abs(ta['perp0', 'L'])**2
                                      - abs(ta['para0', 'R'])**2 - abs(ta['para0', 'L'])**2 )
    K['33'] = 0.25 * alpha * Plb * (abs(ta['perp1', 'R'])**2 + abs(ta['perp1', 'L'])**2
                                  - abs(ta['para1', 'R'])**2 - abs(ta['para1', 'L'])**2 )
    K['34'] = - alpha * K12K34.imag

    return K



def get_ff(q2, par):
    ff_aux = AuxiliaryQuantity['Lambdab->Lambda form factor']
    return ff_aux.prediction(par_dict=par, wc_obj=None, q2=q2)

def prefactor(q2, par, scale):
    xi_t = flavio.physics.ckm.xi('t','bs')(par)
    alphaem = flavio.physics.running.running.get_alpha(par, scale)['alpha_e']
    mLb = par['m_Lambdab']
    mL = par['m_Lambda']
    la_K = flavio.physics.bdecays.common.lambda_K(mLb**2, mL**2, q2)
    return par['GF'] * xi_t * alphaem * sqrt(q2) * la_K**(1/4.) / sqrt(3 * 2 * mLb**3 * pi**5) / 32.


def get_transverity_amps_ff(q2, wc_obj, par_dict, lep, cp_conjugate):
    par = par_dict.copy()
    if cp_conjugate:
        par = conjugate_par(par)
    scale = flavio.config['renormalization scale']['lambdab']
    mLb = par['m_Lambdab']
    mL = par['m_Lambda']
    mb = flavio.physics.running.running.get_mb(par, scale)
    ff = get_ff(q2, par)
    wc = flavio.physics.bdecays.wilsoncoefficients.wctot_dict(wc_obj, 'bs' + lep + lep, scale, par)
    wc_eff = flavio.physics.bdecays.wilsoncoefficients.get_wceff(q2, wc, par, 'Lambdab', 'Lambda', lep, scale)
    ha = helicity_amps(q2, mLb, mL, ff)
    N = prefactor(q2, par, scale)
    ta_ff = transverity_amps(ha, q2, mLb, mL, mb, 0, wc_eff, N)
    return ta_ff


# get subleading hadronic contribution at low q2
def get_subleading(q2, wc_obj, par_dict, cp_conjugate):
    if q2 <= 9:
        sub_name = 'Lambdab->Lambdall subleading effects at low q2'
        return AuxiliaryQuantity[sub_name].prediction(par_dict=par_dict, wc_obj=wc_obj, q2=q2, cp_conjugate=cp_conjugate)
    elif q2 > 14:
        sub_name = 'Lambdab->Lambdall subleading effects at high q2'
        return AuxiliaryQuantity[sub_name].prediction(par_dict=par_dict, wc_obj=wc_obj, q2=q2, cp_conjugate=cp_conjugate)
    else:
        return {}

def get_transverity_amps(q2, wc_obj, par, lep, cp_conjugate):
    if q2 >= 8.7 and q2 < 14:
        warnings.warn("The predictions in the region of narrow charmonium resonances are not meaningful")
    return add_dict((
        get_transverity_amps_ff(q2, wc_obj, par, lep, cp_conjugate),
        get_subleading(q2, wc_obj, par, cp_conjugate)
        ))


def get_obs(function, q2, wc_obj, par, lep):
    ml = par['m_'+lep]
    mLb = par['m_Lambdab']
    mL = par['m_Lambda']
    if q2 < 4*ml**2 or q2 > (mLb-mL)**2:
        return 0
    ta = get_transverity_amps(q2, wc_obj, par, lep, cp_conjugate=False)
    alpha = par['Lambda->ppi alpha_-']
    pLb = par['Lambdab polarisation']
    K = angular_coefficients(ta, alpha, pLb)
    return function(K)

def K_num(K, i):
    return K[i]

def dGdq2(K):
    return 2*K['1ss'] + K['1cc']

def FL_num(K):
    return 2*K['1ss'] - K['1cc']

def AFBl_num(K):
    return (3/2.) * K['1c']

def AFBh_num(K):
    return K['2ss'] + K['2cc']/2.

def AFBlh_num(K):
    return (3/4.) * K['2c']

def dbrdq2(q2, wc_obj, par, lep):
    tauLb = par['tau_Lambdab']
    return tauLb * get_obs(dGdq2, q2, wc_obj, par, lep)

def dbrdq2_int(q2min, q2max, wc_obj, par, lep):
    def obs(q2):
        return dbrdq2(q2, wc_obj, par, lep)
    return flavio.math.integrate.nintegrate(obs, q2min, q2max)/(q2max-q2min)

def obs_int(function, q2min, q2max, wc_obj, par, lep):
    def obs(q2):
        return get_obs(function, q2, wc_obj, par, lep)
    return flavio.math.integrate.nintegrate(obs, q2min, q2max)

# Functions returning functions needed for Prediction instances

def dbrdq2_int_func(lep):
    def fct(wc_obj, par, q2min, q2max):
        return dbrdq2_int(q2min, q2max, wc_obj, par, lep)
    return fct

def dbrdq2_func(lep):
    def fct(wc_obj, par, q2):
        return dbrdq2(q2, wc_obj, par, lep)
    return fct

def obs_ratio_func(func_num, func_den, lep):
    def fct(wc_obj, par, q2):
        num = get_obs(func_num, q2, wc_obj, par, lep)
        if num == 0:
            return 0
        denom = get_obs(func_den, q2, wc_obj, par, lep)
        return num/denom
    return fct

def obs_int_ratio_func(func_num, func_den, lep):
    def fct(wc_obj, par, q2min, q2max):
        num = obs_int(func_num, q2min, q2max, wc_obj, par, lep)
        if num == 0:
            return 0
        denom = obs_int(func_den, q2min, q2max, wc_obj, par, lep)
        return num/denom
    return fct


_tex = {'e': 'e', 'mu': '\mu', 'tau': r'\tau'}
_observables = {
'FL': {'func_num': FL_num, 'tex': r'F_L', 'desc': 'longitudinal polarization fraction'},
'AFBl': {'func_num': AFBl_num, 'tex': r'A_\text{FB}^\ell', 'desc': 'leptonic forward-backward asymmetry'},
'AFBh': {'func_num': AFBh_num, 'tex': r'A_\text{FB}^h', 'desc': 'hadronic forward-backward asymmetry'},
'AFBlh': {'func_num': AFBlh_num, 'tex': r'A_\text{FB}^{\ell h}', 'desc': 'lepton-hadron forward-backward asymmetry'},
}
for i in ['1ss', '1cc', '1c', '2ss', '2cc', '2c', '3sc', '3s', '4sc', '4s']:
    _observables[f'K{i}'] = {'func_num': lambda K, i=i: K_num(K, i), 'tex': r'K_{'+i+'}', 'desc': 'angular coefficient '+i}
for i in range(11, 35):
    _observables[f'K{i}'] = {'func_num': lambda K, i=i: K_num(K, str(i)), 'tex': r'K_{'+str(i)+'}', 'desc': 'angular coefficient '+str(i)}
# for i in range(1, 35):
#     _observables[f'K{i}'] = {'func_num': lambda K, i=i: K_num(K, i), 'tex': r'K_{'+str(i)+'}', 'desc': 'angular coefficient '+str(i)}

for l in ['e', 'mu', ]: # tau requires lepton mass dependence!

    _process_tex = r"\Lambda_b\to\Lambda " +_tex[l]+r"^+"+_tex[l]+r"^-"
    _process_taxonomy = r'Process :: $b$ hadron decays :: FCNC decays :: $\Lambda_b\to \Lambda\ell^+\ell^-$ :: $' + _process_tex + r"$"

    # binned branching ratio
    _obs_name = "<dBR/dq2>(Lambdab->Lambda"+l+l+")"
    _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
    _obs.set_description(r"Binned differential branching ratio of $" + _process_tex + r"$")
    _obs.tex = r"$\langle \frac{d\text{BR}}{dq^2} \rangle(" + _process_tex + r")$"
    _obs.add_taxonomy(_process_taxonomy)
    Prediction(_obs_name, dbrdq2_int_func(l))

    # differential branching ratio
    _obs_name = "dBR/dq2(Lambdab->Lambda"+l+l+")"
    _obs = Observable(name=_obs_name, arguments=['q2'])
    _obs.set_description(r"Differential branching ratio of $" + _process_tex + r"$")
    _obs.tex = r"$\frac{d\text{BR}}{dq^2}(" + _process_tex + r")$"
    _obs.add_taxonomy(_process_taxonomy)
    Prediction(_obs_name, dbrdq2_func(l))

    for obs in _observables:
        # binned angular observables
        _obs_name = "<" + obs + ">(Lambdab->Lambda"+l+l+")"
        _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
        _obs.set_description("Binned " + _observables[obs]['desc'] + r" in $" + _process_tex + r"$")
        _obs.tex = r"$\langle " + _observables[obs]['tex'] + r"\rangle(" + _process_tex + r")$"
        _obs.add_taxonomy(_process_taxonomy)
        Prediction(_obs_name, obs_int_ratio_func(_observables[obs]['func_num'], dGdq2, l))

        # differential angular observables
        _obs_name = obs + "(Lambdab->Lambda"+l+l+")"
        _obs = Observable(name=_obs_name, arguments=['q2'])
        _obs.set_description(_observables[obs]['desc'][0].capitalize() + _observables[obs]['desc'][1:] + r" in $" + _process_tex + r"$")
        _obs.tex = r"$" + _observables[obs]['tex'] + r"(" + _process_tex + r")$"
        _obs.add_taxonomy(_process_taxonomy)
        Prediction(_obs_name, obs_ratio_func(_observables[obs]['func_num'], dGdq2, l))
