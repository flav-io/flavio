r"""Functions for $\Lambda_b \to \Lambda(1520)(\to NK) \ell^+ \ell⁻$ decays as in arXiv:1903.00448."""

import flavio
from math import sqrt, pi
from flavio.physics.bdecays.common import lambda_K, beta_l, meson_quark, meson_ff
from flavio.classes import Observable, Prediction, AuxiliaryQuantity
from flavio.physics.common import conjugate_par, conjugate_wc, add_dict
import warnings


def helicity_amps(q2, mLb, mL, ff):
    # Hadronic helicity amplitudes of Lb->L(1520)(->pK)l+l-
    sp = (mLb + mL)**2 - q2
    sm = (mLb - mL)**2 - q2

    H = {}
    # non-vanishing amplitudes in the vector part (V)
    # eqs (3.8) of arXiv:1903.00448
    # 1 for 1/2, 3 for 3/2
    H['tV+1+1'] = ff['fVt'] * (mLB - mL)/sqrt(q2) * (sp * sqrt(sm))/(sqrt(6) * mL)
    H['tV-1-1'] = H['tV+1+1']
    H['0V+1+1'] = -ff['fV0'] * (mLB + mL)/sqrt(q2) * (sm * sqrt(sp))/(sqrt(6) * mL)
    H['0V-1-1'] = H['0V+1+1']
    H['+V+1-1'] = -ff['fVperp'] * (sm * sqrt(sp))/(sqrt(3) * mL)
    H['-V-1+1'] = H['+V+1-1']
    H['+V-1-3'] = ff['fVg'] * sqrt(sp)
    H['-V+1+3'] = H['+V-1-3']

    # for the axial part (A), eqs (3.9)
    H['tA+1+1'] = ff['fAt'] * (mLB + mL)/sqrt(q2) * (sm * sqrt(sp))/(sqrt(6) * mL)
    H['tA-1-1'] = -H['tA+1+1']
    H['0A+1+1'] = -ff['fA0'] * (mLB - mL)/sqrt(q2) * (sp * sqrt(sm))/(sqrt(6) * mL)
    H['0A-1-1'] = -H['0A+1+1']
    H['+A+1-1'] = ff['fAperp'] * (sp * sqrt(sm))/(sqrt(3) * mL)
    H['-A-1+1'] = -H['+A+1-1']
    H['+A-1-3'] = -ff['fAg'] * sqrt(sm)
    H['-A+1+3'] = -H['+A-1-3']

    # for the tensor part (T), eqs (3.15)
    H['0T+1+1'] = ff['fT0'] * sqrt(q2) * (sm * sqrt(sp))/(sqrt(6) * mL)
    H['0T-1-1'] = H['0T+1+1']
    H['+T+1-1'] = ff['fTperp'] * (mLB + mL) * (sm * sqrt(sp))/(sqrt(3) * mL)
    H['-T-1+1'] = H['+T+1-1']
    H['+T-1-3'] = -ff['fTg'] * sqrt(sp) 
    H['-T+1+3'] = H['+T-1-3']

    H['0T5+1+1'] = -ff['fT50'] * sqrt(q2) * (sp * sqrt(sm))/(sqrt(6) * mL)
    H['0T5-1-1'] = -H['0T5+1+1']
    H['+T5+1-1'] = ff['fT5perp'] * (mLB - mL) * (sp * sqrt(sm))/(sqrt(3) * mL)
    H['-T5-1+1'] = -H['+T5+1-1']
    H['+T5-1-3'] = -ff['fT5g'] * sqrt(sm) 
    H['-T5+1+3'] = -H['+T5-1-3']

    return H


def transversity_amps(ha, q2, mLb, mL, mqh, wc, prefactor):
    # Hadronic transversity amplitudes
    # defined as in eqs (3.18) and (3.20)

    C910Lpl = (wc['v'] - wc['a']) + (wc['vp'] - wc['ap'])
    C910Rpl = (wc['v'] + wc['a']) + (wc['vp'] + wc['ap'])
    C910Lmi = (wc['v'] - wc['a']) - (wc['vp'] - wc['ap'])
    C910Rmi = (wc['v'] + wc['a']) - (wc['vp'] + wc['ap'])

    A = {}

    A['Bperp1', 'L'] = sqrt(2)*( C910Lpl * ha['+V-1-3'] - 2*mqh*(wc['7']+wc['7p'])/q2 * ha['+T-1-3'])
    A['Bperp1', 'R'] = sqrt(2)*( C910Rpl * ha['+V-1-3'] - 2*mqh*(wc['7']+wc['7p'])/q2 * ha['+T-1-3'])
    A['Bpara1', 'L'] = -sqrt(2)*( C910Lmi * ha['+A-1-3'] + 2*mqh*(wc['7']-wc['7p'])/q2 * ha['+T5-1-3'])
    A['Bpara1', 'R'] = -sqrt(2)*( C910Rmi * ha['+A-1-3'] + 2*mqh*(wc['7']-wc['7p'])/q2 * ha['+T5-1-3'])

    
    A['Aperp1', 'L'] = sqrt(2)*( C910Lpl * ha['+V+1-1'] - 2*mqh*(wc['7']+wc['7p'])/q2 * ha['+T+1-1'])
    A['Aperp1', 'R'] = sqrt(2)*( C910Rpl * ha['+V+1-1'] - 2*mqh*(wc['7']+wc['7p'])/q2 * ha['+T+1-1'])
    A['Apara1', 'L'] = -sqrt(2)*( C910Lmi * ha['+A+1-1'] + 2*mqh*(wc['7']-wc['7p'])/q2 * ha['+T5+1-1'])
    A['Apara1', 'R'] = -sqrt(2)*( C910Rmi * ha['+A+1-1'] + 2*mqh*(wc['7']-wc['7p'])/q2 * ha['+T5+1-1'])

    A['Aperp0', 'L'] = sqrt(2)*( C910Lpl * ha['0V+1+1'] - 2*mqh*(wc['7']+wc['7p'])/q2 * ha['0T+1+1'])
    A['Aperp0', 'R'] = sqrt(2)*( C910Rpl * ha['0V+1+1'] - 2*mqh*(wc['7']+wc['7p'])/q2 * ha['0T+1+1'])
    A['Apara0', 'L'] = -sqrt(2)*( C910Lmi * ha['0A+1+1'] + 2*mqh*(wc['7']-wc['7p'])/q2 * ha['0T5+1+1'])
    A['Apara0', 'R'] = -sqrt(2)*( C910Rmi * ha['0A+1+1'] + 2*mqh*(wc['7']-wc['7p'])/q2 * ha['0T5+1+1'])

    return {k: prefactor*v for k, v in A.items()}


def angular_coefficients(ta, br):
    # eqs (4.2) in arxiv
    # br is BR(L(1520)->pK)

    L={}
    L['1c'] = -2*br*( (ta['Aperp1','L'] * ta['Apara1','L'].conj()).real
                      - (ta['Aperp1','R'] * ta['Apara1','R'].conj()).real
    )
    L['1cc'] = br*( abs(ta['Apara1','L'])**2 + abs(ta['Aperp1','L'])**2
                    + abs(ta['Apara1','R'])**2 + abs(ta['Aperp1','R'])**2
    )
    L['1ss'] = br/2*( 2*(abs(ta['Apara0','L'])**2 + abs(ta['Aperp0','L'])**2)
                      + abs(ta['Apara1','L'])**2 + abs(ta['Aperp1','L'])**2
                      + 2*(abs(ta['Apara0','R'])**2 + abs(ta['Aperp0','R'])**2)
                      + abs(ta['Apara1','R'])**2 + abs(ta['Aperp1','R'])**2
    )
    L['2c'] = -br/2*( (ta['Aperp1','L'] * ta['Apara1','L'].conj()).real
                      + 3*(ta['Bperp1','L'] * ta['Bpara1','L'].conj()).real
                      - (ta['Aperp1','R'] * ta['Apara1','R'].conj()).real
                      - 3*(ta['Bperp1','R'] * ta['Bpara1','R'].conj()).real
    )
    L['2cc'] = br/4*( abs(ta['Apara1','L'])**2 + abs(ta['Aperp1','L'])**2
                      + 3*(abs(ta['Bpara1','L'])**2 + abs(ta['Bperp1','L'])**2)
                      + abs(ta['Apara1','R'])**2 + abs(ta['Aperp1','R'])**2
                      + 3*(abs(ta['Bpara1','R'])**2 + abs(ta['Bperp1','R'])**2)
    )
    L['2ss'] = br/8*( 2*abs(ta['Apara0','L'])**2 + abs(ta['Apara1','L'])**2
                      + 2*abs(ta['Aperp0','L'])**2 + abs(ta['Aperp1','L'])**2
                      + 3*(abs(ta['Bpara1','L'])**2 + abs(ta['Bperp1','L'])**2) 
                      - 2*sqrt(3)*(ta['Bpara1','L']*ta['Apara1','L'].conj()).real
                      + 2*sqrt(3)*(ta['Bperp1','L']*ta['Aperp1','L'].conj()).real
                      + 2*abs(ta['Apara0','R'])**2 + abs(ta['Apara1','R'])**2
                      + 2*abs(ta['Aperp0','R'])**2 + abs(ta['Aperp1','R'])**2
                      + 3*(abs(ta['Bpara1','R'])**2 + abs(ta['Bperp1','R'])**2) 
                      - 2*sqrt(3)*(ta['Bpara1','R']*ta['Apara1','R'].conj()).real
                      + 2*sqrt(3)*(ta['Bperp1','R']*ta['Aperp1','R'].conj()).real
    )
    L['3ss'] = sqrt(3)/2*br*( (ta['Bpara1','L']*ta['Apara1','L'].conj()).real
                              - (ta['Bperp1','L']*ta['Aperp1','L'].conj()).real
                              + (ta['Bpara1','R']*ta['Apara1','R'].conj()).real
                              - (ta['Bperp1','R']*ta['Aperp1','R'].conj()).real 
    )
    L['4ss'] = sqrt(3)/2*br*( (ta['Bperp1','L']*ta['Apara1','L'].conj()).imag
                              - (ta['Bpara1','L']*ta['Aperp1','L'].conj()).imag
                              + (ta['Bperp1','R']*ta['Apara1','R'].conj()).imag
                              - (ta['Bpara1','R']*ta['Aperp1','R'].conj()).imag
    )
    L['5s'] = sqrt(3/2)*br*( (ta['Bperp1','L']*ta['Apara0','L'].conj()).real
                             - (ta['Bpara1','L']*ta['Aperp0','L'].conj()).real
                             - (ta['Bperp1','R']*ta['Apara0','R'].conj()).real
                             + (ta['Bpara1','R']*ta['Aperp0','R'].conj()).real
    ) 
    L['5sc'] = sqrt(3/2)*br*( -(ta['Bpara1','L']*ta['Apara0','L'].conj()).real
                              + (ta['Bperp1','L']*ta['Aperp0','L'].conj()).real
                              - (ta['Bpara1','R']*ta['Apara0','R'].conj()).real
                              + (ta['Bperp1','R']*ta['Aperp0','R'].conj()).real
    )
    L['6s'] = sqrt(3/2)*br*( (ta['Bpara1','L']*ta['Apara0','L'].conj()).imag
                             - (ta['Bperp1','L']*ta['Aperp0','L'].conj()).imag
                             - (ta['Bpara1','R']*ta['Apara0','R'].conj()).imag
                             + (ta['Bperp1','R']*ta['Aperp0','R'].conj()).imag
    ) 
    L['6sc'] = -sqrt(3/2)*br*( (ta['Bperp1','L']*ta['Apara0','L'].conj()).imag
                             - (ta['Bpara1','L']*ta['Aperp0','L'].conj()).imag
                             + (ta['Bperp1','R']*ta['Apara0','R'].conj()).imag
                             - (ta['Bpara1','R']*ta['Aperp0','R'].conj()).imag
    )

    return L
    

def prefactor(q2, par, scale):
    # calculate prefactor N
    xi_t = flavio.physics.ckm.xi('t','bs')(par)
    alphaem = flavio.physics.running.running.get_alpha(par, scale)['alpha_e']
    mLb = par['m_Lambdab']
    mL = par['m_Lambda(1520)']
    la_K = flavio.physics.bdecays.common.lambda_K(mLb**2, mL**2, q2)
    return par['GF'] * xi_t * alphaem * sqrt(q2) * la_K**(1/4.) / sqrt(3 * 2 * mLb**3 * pi**5) / 32


# !!! form factors L -> L(1520) !!!
def get_ff(q2, par):
    ff_aux = AuxiliaryQuantity['Lambdab->Lambda(1520) form factor']
    return ff_aux.prediction(par_dict=par, wc_obj=None, q2=q2)


# !!! get subleading hadronic contribution at low q2 !!!
def get_subleading(q2, wc_obj, par_dict, cp_conjugate):
    if q2 <= 9:
        subname = 'Lambdab->Lambda(1520)ll subleading effects at low q2'
        return AuxiliaryQuantity[subname].prediction(par_dict=par_dict, wc_obj=wc_obj, q2=q2, cp_conjugate=cp_conjugate)
    elif q2 > 14:
        subname = 'Lambdab->Lambda(1520)ll subleading effects at high q2'
        return AuxiliaryQuantity[subname].prediction(par_dict=par_dict, wc_obj=wc_obj, q2=q2, cp_conjugate=cp_conjugate)
    else:
        return {}


def get_transversity_amps_ff(q2, wc_obj, par_dict, lep, cp_conjugate):
    par = par_dict.copy()
    if cp_conjugate:
        par = conjugate_par(par)
    scale = flavio.config['renormalization scale']['lambdab']
    mLb = par['m_Lambdab']
    mL = par['m_Lambda(1520)']
    mb = flavio.physics.running.running.get_mb(par, scale)
    ff = get_ff(q2, par)
    wc = flavio.physics.bdecays.wilsoncoefficients.wctot_dict(wc_obj, 'bs' + lep + lep, scale, par)
    # Is wc_eff working correctly ? 
    wc_eff = flavio.physics.bdecays.wilsoncoefficients.get_wceff(q2, wc, par, 'Lambdab', 'Lambda(1520)', lep, scale)
    ha = helicity_amps(q2, mLb, mL, ff)
    N = prefactor(q2, par, scale)
    ta_ff = transversity_amps(ha, q2, mLb, mL, mb, 0, wc_eff, N)
    return ta_ff


def get_transversity_amps(q2, wc_obj, par, lep, cp_conjugate):
    if q2 >= 8.7 and q2 < 14:
        warnings.warn("The prediction in the region of narrow charmonium resonances are not meaningful")
    return add_dict((
        get_transversity_ams_ff(q2, wc_obj, par, lep, cp_conjugate),
        get_subleading(q2, wc_ovj, par, cp_conjugate)
        ))


def get_obs(function, q2, wc_obj, par, lep):
    ml = par['m_'+lep]
    mLb = par['m_Lambdab']
    mL = par['m_Lambda(1520)']
    if q2 < 4*ml**2 or q2 > (mLb-mL)**2:
        return 0
    ta = get_transversity_amps(q2, wc_obj, par, lep, cp_conjugate=False)
    BR = par['BR(Lambda(1520)->NKbar)_exp']/2
    L = angular_coefficients(ta, BR)
    return function(L)


def get_obs_new(function, q2, wc_obj, par, lep, arg):
    ml = par['m_'+lep]
    mLb = par['m_Lambdab']
    mL = par['m_Lambda(1520)']
    if q2 < 4*ml**2 or q2 > (mLb-mL)**2:
        return 0
    ta = get_transversity_amps(q2, wc_obj, par, lep, cp_conjugate=False)
    ta_conj = get_transversity_amps(q2, wc_obj, par, lep, cp_conjugate=True)    
    BR = par['BR(Lambda(1520)->NKbar)_exp']/2
    L = angular_coefficients(ta, BR)
    dG = dGdq2(L)
    L_conj = angular_coefficients(ta, BR)
    dG_conj = dGdq2(L_conj)
    return function(L, L_conj, dG, dG_conj, arg)


# OBSERVABLES
def dGdq2(L):
    # differential decay width
    return [L['1cc'] + 2*L['1ss'] + 2*L['2cc'] + 4*L['2ss'] + 2*L['3ss']]/3


def S(L, L_conj, dG, dG_conj, arg):
    # CP-averaged angular observalbes
    # arg is for example '1cc'
    if L[arg] + L_conj[arg] == 0:
        return 0
    else:
        return ( L[arg] + L_conj[arg] )/( dG + dG_conj  ) 


def A(L, L_conj, dG, dG_conj, arg):
    # CP-asymmetries
    # arg is for example '1cc'
    if L[arg] - L_conj[arg] == 0:
        return 0
    else:
        return ( L[arg] - L_conj[arg] )/( dG + dG_conj  ) 


def FL_num(L):
    # longuitudinal polarization of the dilepton system
    return 1-2*(L['1cc'] + 2*L['2cc'])/(3)


def AFBl_num(L):
    return (L['1c'] + 2*L['2c'])/(2)


def AFBh:
    return 0


def AFBlh:
    return 0


def dbrdq2(q2, wc_obj, par, lep):
    tauLb = par['tau_Lambdab']
    return tauLb * get_obs(dGdq2, q2, wc_obj, par, lep)


def dbrdq2_int(q2min, q2max, wc_obj, par, lep):
    def obs(q2):
        return dbrdq2(q2, wc_obj, par, lep)
    return flavio.math.integrate.nintegrate(obs, q2min, q2max)/(q2max-q2min)


def obs_int(function, q2min, q2max, wc_obj, par, lep):
    def obs(q2):
        return get_obs(function, q2, wc_obj, par, lep):
    return flavio.math.integrate.nintegrate(obs, q2min, q2max)


def obs_int_new(function, q2min, q2max, wc_obj, par, lep, arg):
    def obs(q2):
        return get_obs_new(function, q2, wc_obj, par, lep, arg):
    return flavio.math.integrate.nintegrate(obs, q2min, q2max)


# Functions returning functions needed for Prediction instance

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


def obs_ratio_func_new(func, lep, arg):
    def fct(wc_obj, par, q2):
        return get_obs_new(func, q2, wc_obj, par, lep, arg)
    return fct


def obs_int_ratio_func(func_num, func_den, lep):
    def fct(wc_obj, par, q2min, q2max):
        num = obs_int(func_num, q2min, q2max, wc_obj, par, lep)
        if num == 0:
            return 0
        denom = obs_int(func_den, q2min, q2max, wc_obj, par, lep)
        return num/denom
    return fct


def obs_int_ratio_func_new(func, lep, arg):
    def fct(wc_obj, par, q2min, q2max):
        return obs_int_new(func, q2, wc_obj, par, lep, arg)
    return fct


_tex = {'e': 'e', 'mu': r'\mu', 'tau': r'\tau'}
_observables = {
    'FL': {'func_num': FL_num, 'tex': r'F_L'. 'desc': 'longitudinal polarization fraction'},
    'AFBl': {'func_num': AFBl_num, 'tex': r'A_\text{FB}^\ell', 'desc': 'leptonic forward-backward asymmetry'},
    'AFBh': {'func_num': AFBh, 'tex': r'A_\text{FB}^\ell', 'desc': 'hadronic forward-backward asymmetry'},
    'AFBlh': {'func_num': AFBlh, 'tex': r'A_\text{FB}^{\ell h}', 'desc': 'lepton-hadron forward-backward asymmetry'}
    }


arg_List = ['1c', '1cc', '1ss', '2c', '2cc', '2ss', '3ss', '4ss', '5s', '5sc', '6s', '6sc']

_observables_new = {}
for a in arg_List:
    S_string = 'S_'+a
    A_string = 'A_'+a
    _observables_new[S_string] = {'func': S, 'tex': r'S_{'+a+'}', 'desc': 'CP symmetry '+a, 'arg': a}
    _observables_new[A_string] = {'func': A, 'tex': r'A_{'+a+'}', 'desc': 'CP asymmetry '+a, 'arg': a}

    
for l in ['e', 'mu']:

    _process_tex = r'\Lambda_b\to\Lambda(1520) '+_tex[l]+r'^+'+_tex[l]+r'^-'
    _process_taxonomy = r'Process :: $b$ hadron decays :: FCNC :: $\Lambda_b\to\Lambda(1520)\ell^+\ell^-$ :: $' + _process_tex + r'$'

    # binned branching ratio
    _obs_name = "<dBR/q2>(Lambdab->Lambda"+l+l+")"
    _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
    _obs.set_description(r"Binned differential branching ratio of $" + _process_tex + r"$")
    _obs.tex = r"$\langle \frac{d\text{BR}}{dq^2} \rangle(" + _process_tex + r")$"
    _obs.add_taxomony(_process_taxonomy)
    Prediction(_obs_name, dbrdq2_int_func(l))

    # differential branching ratio
    _obs_name = "dBR/q2(Lambdab->Lambda"+l+l+")"
    _obs = Observable(name=_obs_name, arguments=['q2'])
    _obs.set_description(r"Differential branching ratio of $" + _process_tex + r"$")
    _obs.tex = r"$\frac{d\text{BR}}{dq^2}(" + _process_tex + r")$"
    _obs.add_taxomony(_process_taxonomy)
    Prediction(_obs_name, dbrdq2_func(l))

    for obs in _observables:
        # binned angular observables
        _obs_name = "<" + obs ">(Lambdab->Lambda"+l+l+")"
        _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
        _obs.set_description("Binned " + _observables[obs]['desc'] + r" in $" + _process_tex + r"$")
        _obs.tex = r"$\langle " + _observables[obs]['tex'] + r"\rangle(" + _process_tex + r"$"
        _obs.add_taxonomy(_proces_taxonomy)
        Prediction(_obs_name, obs_int_ratio_func(_observables[obs]['func_num'], dGdq2, l))

        # differential angular observables
        _obs_name = obs "(Lambdab->Lambda"+l+l+")"
        _obs = Observable(name=_obs_name, arguments=['q2'])
        _obs.set_description(_observables[obs]['desc'][0].capitalize() + _observables[obs]['desc'][1:] r" in $" + _process_tex + r"$")
        _obs.tex = r"$" + _observables[obs]['tex'] + r"(" + _process_tex + r"$"
        _obs.add_taxonomy(_proces_taxonomy)
        Prediction(_obs_name, obs_ratio_func(_observables[obs]['func_num'], dGdq2, l))

        
    # Adding CP-symmetries and asymmetries
    for obs in _observables_new:
        # binned angular observables
        _obs_name = "<" + obs ">(Lambdab->Lambda"+l+l+")"
        _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
        _obs.set_description("Binned " + _observables[obs]['desc'] + r" in $" + _process_tex + r"$")
        _obs.tex = r"$\langle " + _observables[obs]['tex'] + r"\rangle(" + _process_tex + r"$"
        _obs.add_taxonomy(_proces_taxonomy)
        Prediction(_obs_name, obs_int_ratio_func_new(_observables[obs]['func'], dGdq2, l, _observables[obs]['arg']))

        # differential angular observables
        _obs_name = obs "(Lambdab->Lambda"+l+l+")"
        _obs = Observable(name=_obs_name, arguments=['q2'])
        _obs.set_description(_observables[obs]['desc'][0].capitalize() + _observables[obs]['desc'][1:] r" in $" + _process_tex + r"$")
        _obs.tex = r"$" + _observables[obs]['tex'] + r"(" + _process_tex + r"$"
        _obs.add_taxonomy(_proces_taxonomy)
        Prediction(_obs_name, obs_ratio_func_new(_observables[obs]['func'], dGdq2, l, _observables[obs]['arg']))
    
