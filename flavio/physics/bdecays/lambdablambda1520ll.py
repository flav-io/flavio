r"""Functions for $\Lambda_b \to \Lambda(1520)(\to NK) \ell^+ \ell⁻$ decays as in arXiv:1903.00448."""

import flavio
from math import sqrt, pi
from flavio.physics.bdecays.common import lambda_K, beta_l, meson_quark, meson_ff
from flavio.classes import Observable, Prediction, AuxiliaryQuantity
from flavio.physics.common import conjugate_par, conjugate_wc, add_dict
import warnings


def helicity_amps(q2, mLb, mL, ff):
    flavio.citations.register("Descotes-Genon:2019dbw")

    # Hadronic helicity amplitudes of Lb->L(1520)(->pK)l+l-
    sp = (mLb + mL)**2 - q2
    sm = (mLb - mL)**2 - q2

    H = {}
    # non-vanishing amplitudes in the vector part (V)
    # eqs (3.8) of arXiv:1903.00448
    # 1 for 1/2, 3 for 3/2
    H['tV+1+1'] = ff['fVt'] * (mLb - mL)/sqrt(q2) * (sp * sqrt(sm))/(sqrt(6) * mL)
    H['tV-1-1'] = H['tV+1+1']
    H['0V+1+1'] = -ff['fV0'] * (mLb + mL)/sqrt(q2) * (sm * sqrt(sp))/(sqrt(6) * mL)
    H['0V-1-1'] = H['0V+1+1']
    H['+V+1-1'] = -ff['fVperp'] * (sm * sqrt(sp))/(sqrt(3) * mL)
    H['-V-1+1'] = H['+V+1-1']
    H['+V-1-3'] = ff['fVg'] * sqrt(sp)
    H['-V+1+3'] = H['+V-1-3']

    # for the axial part (A), eqs (3.9)
    H['tA+1+1'] = ff['fAt'] * (mLb + mL)/sqrt(q2) * (sm * sqrt(sp))/(sqrt(6) * mL)
    H['tA-1-1'] = -H['tA+1+1']
    H['0A+1+1'] = -ff['fA0'] * (mLb - mL)/sqrt(q2) * (sp * sqrt(sm))/(sqrt(6) * mL)
    H['0A-1-1'] = -H['0A+1+1']
    H['+A+1-1'] = ff['fAperp'] * (sp * sqrt(sm))/(sqrt(3) * mL)
    H['-A-1+1'] = -H['+A+1-1']
    H['+A-1-3'] = -ff['fAg'] * sqrt(sm)
    H['-A+1+3'] = -H['+A-1-3']

    # for the tensor part (T), eqs (3.15)
    H['0T+1+1'] = ff['fT0'] * sqrt(q2) * (sm * sqrt(sp))/(sqrt(6) * mL)
    H['0T-1-1'] = H['0T+1+1']
    H['+T+1-1'] = ff['fTperp'] * (mLb + mL) * (sm * sqrt(sp))/(sqrt(3) * mL)
    H['-T-1+1'] = H['+T+1-1']
    H['+T-1-3'] = -ff['fTg'] * sqrt(sp)
    H['-T+1+3'] = H['+T-1-3']

    H['0T5+1+1'] = -ff['fT50'] * sqrt(q2) * (sp * sqrt(sm))/(sqrt(6) * mL)
    H['0T5-1-1'] = -H['0T5+1+1']
    H['+T5+1-1'] = ff['fT5perp'] * (mLb - mL) * (sp * sqrt(sm))/(sqrt(3) * mL)
    H['-T5-1+1'] = -H['+T5+1-1']
    H['+T5-1-3'] = -ff['fT5g'] * sqrt(sm)
    H['-T5+1+3'] = -H['+T5-1-3']

    return H


def transversity_amps(ha, q2, mLb, mL, mqh, wc, prefactor, t):
    # Hadronic transversity amplitudes
    # defined as in eqs (3.18) and (3.20)

    ## by taking wc_eff
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
    la_k = flavio.physics.bdecays.common.lambda_K(mLb**2, mL**2, q2)
    return par['GF'] * xi_t * alphaem * sqrt(q2) * la_k**(1/4.) / sqrt(3 * 2 * mLb**3 * pi**5) / 32


def get_ff(q2, par):
    ff_aux = AuxiliaryQuantity['Lambdab->Lambda(1520) form factor']
    return ff_aux.prediction(par_dict=par, wc_obj=None, q2=q2)


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
    wc_eff = flavio.physics.bdecays.wilsoncoefficients.get_wceff(q2, wc, par, 'Lambdab', 'Lambda(1520)', lep, scale)

    ha = helicity_amps(q2, mLb, mL, ff)
    N = prefactor(q2, par, scale)
    ta_ff = transversity_amps(ha, q2, mLb, mL, mb, wc_eff, N, 'bs'+lep+lep)
    return ta_ff


def get_transversity_amps(q2, wc_obj, par, lep, cp_conjugate):
    if q2 >= 8.7 and q2 < 14:
        warnings.warn("The prediction in the region of narrow charmonium resonances are not meaningful")
    return get_transversity_amps_ff(q2, wc_obj, par, lep, cp_conjugate)


def get_obs(function, q2, wc_obj, par, lep, *ang_coef):
    ml = par['m_'+lep]
    mLb = par['m_Lambdab']
    mL = par['m_Lambda(1520)']
    if q2 < 4*ml**2 or q2 > (mLb-mL)**2:
        return 0
    ta = get_transversity_amps(q2, wc_obj, par, lep, cp_conjugate=False)

    br = par['BR(Lambda(1520)->NKbar)_exp']/2
    L = angular_coefficients(ta, br)
    ta_conj = get_transversity_amps(q2, wc_obj, par, lep, cp_conjugate=True)
    L_conj = angular_coefficients(ta_conj, br)
    return function(L, L_conj, *ang_coef)


# OBSERVABLES
def dGdq2(L, L_conj, *args):
    # differential decay width : (dG + dGbar)/dq2*(1/2)
    return (L['1cc'] + 2*L['1ss'] + 2*L['2cc'] + 4*L['2ss'] + 2*L['3ss'] + L_conj['1cc'] + 2*L_conj['1ss'] + 2*L_conj['2cc'] + 4*L_conj['2ss'] + 2*L_conj['3ss'])/6


def S(L, L_conj, ang_coef):
    # CP-averaged angular observalbes
    # ang_coef is for example '1cc'
    return ( L[ang_coef] + L_conj[ang_coef] )/2


def A(L, L_conj, ang_coef):
    # CP-asymmetries
    # ang_coef is for example '1cc'
    return ( L[ang_coef] - L_conj[ang_coef] )/2


def FL_num(L, L_conj, *args):
    # longuitudinal polarization of the dilepton system
    return (-L['1cc'] + 2*L['1ss'] - 2*L['2cc'] + 4*L['2ss'] + 2*L['3ss'] - L_conj['1cc'] + 2*L_conj['1ss'] - 2*L_conj['2cc'] + 4*L_conj['2ss'] + 2*L_conj['3ss'])/6


def AFBl_num(L, L_conj, *args):
    return (L['1c'] + 2*L['2c'] + L_conj['1c'] + 2*L_conj['2c'])/4


def AFBh(*args):
    return 0


def AFBlh(*args):
    return 0


def obs_int(function, q2min, q2max, wc_obj, par, lep, *ang_coef):
    def obs(q2):
        return get_obs(function, q2, wc_obj, par, lep, *ang_coef)
    return flavio.math.integrate.nintegrate(obs, q2min, q2max)


# Functions returning functions needed for Prediction instance

def dbrdq2_int_func(lep):
    def fct(wc_obj, par, q2min, q2max):
        tauLb = par['tau_Lambdab']
        return tauLb*obs_int(dGdq2, q2min, q2max, wc_obj, par, lep)/(q2max-q2min)
    return fct


def dbrdq2_func(lep):
    def fct(wc_obj, par, q2):
        tauLb = par['tau_Lambdab']
        return tauLb * get_obs(dGdq2, q2, wc_obj, par, lep)
    return fct


def obs_ratio_func(func_num, func_den, lep, *ang_coef):
    def fct(wc_obj, par, q2):
        num = get_obs(func_num, q2, wc_obj, par, lep, *ang_coef)
        if num == 0:
            return 0
        denom = get_obs(func_den, q2, wc_obj, par, lep, *ang_coef)
        return num/denom
    return fct


def obs_int_ratio_func(func_num, func_den, lep, *ang_coef):
    def fct(wc_obj, par, q2min, q2max):
        num = obs_int(func_num, q2min, q2max, wc_obj, par, lep, *ang_coef)
        if num == 0:
            return 0
        denom = obs_int(func_den, q2min, q2max, wc_obj, par, lep, *ang_coef)
        return num/denom
    return fct


_tex = {'e': 'e', 'mu': r'\mu', 'tau': r'\tau'}
_observables = {
    'FL': {'func_num': FL_num, 'tex': r'F_L', 'desc': 'longitudinal polarization fraction'},
    'AFBl': {'func_num': AFBl_num, 'tex': r'A_\text{FB}^\ell', 'desc': 'leptonic forward-backward asymmetry'},
    'AFBh': {'func_num': AFBh, 'tex': r'A_\text{FB}^\ell', 'desc': 'hadronic forward-backward asymmetry'},
    'AFBlh': {'func_num': AFBlh, 'tex': r'A_\text{FB}^{\ell h}', 'desc': 'lepton-hadron forward-backward asymmetry'}
    }

# subscript of angular coefficients L
ang_coef_List = ['1c', '1cc', '1ss', '2c', '2cc', '2ss', '3ss', '4ss', '5s', '5sc', '6s', '6sc']

for a in ang_coef_List:
    Sstring = 'S_'+a
    Astring = 'A_'+a
    _observables[Sstring] = {'func_num': S, 'tex': r'S_{'+a+'}', 'desc': 'CP symmetry '+a, 'ang_coef': a}
    _observables[Astring] = {'func_num': A, 'tex': r'A_{'+a+'}', 'desc': 'CP asymmetry '+a, 'ang_coef': a}


for l in ['e', 'mu', ]:

    _process_tex = r'\Lambda_b\to\Lambda(1520) '+_tex[l]+r'^+'+_tex[l]+r'^-'
    _process_taxonomy = r'Process :: $b$ hadron decays :: FCNC decays :: $\Lambda_b\to\Lambda(1520)\ell^+\ell^-$ :: $' + _process_tex + r'$'

    # binned branching ratio
    _obs_name = "<dBR/dq2>(Lambdab->Lambda(1520)"+l+l+")"
    _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
    _obs.set_description(r"Binned differential branching ratio of $" + _process_tex + r"$")
    _obs.tex = r"$\langle \frac{d\text{BR}}{dq^2} \rangle(" + _process_tex + r")$"
    _obs.add_taxonomy(_process_taxonomy)
    Prediction(_obs_name, dbrdq2_int_func(l))

    # differential branching ratio
    _obs_name = "dBR/dq2(Lambdab->Lambda(1520)"+l+l+")"
    _obs = Observable(name=_obs_name, arguments=['q2'])
    _obs.set_description(r"Differential branching ratio of $" + _process_tex + r"$")
    _obs.tex = r"$\frac{d\text{BR}}{dq^2}(" + _process_tex + r")$"
    _obs.add_taxonomy(_process_taxonomy)
    Prediction(_obs_name, dbrdq2_func(l))

    for obs in _observables:
        # binned angular observables
        _obs_name = "<" + obs+">(Lambdab->Lambda(1520)"+l+l+")"
        _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
        _obs.set_description("Binned " + _observables[obs]['desc'] + r" in $" + _process_tex + r"$")
        _obs.tex = r"$\langle " + _observables[obs]['tex'] + r"\rangle(" + _process_tex + r")$"
        _obs.add_taxonomy(_process_taxonomy)
        if 'ang_coef' in _observables[obs].keys():
            Prediction(_obs_name, obs_int_ratio_func(_observables[obs]['func_num'], dGdq2, l, _observables[obs]['ang_coef']))
        else :
            Prediction(_obs_name, obs_int_ratio_func(_observables[obs]['func_num'], dGdq2, l))

        # differential angular observables
        _obs_name = obs+"(Lambdab->Lambda(1520)"+l+l+")"
        _obs = Observable(name=_obs_name, arguments=['q2'])
        _obs.set_description(_observables[obs]['desc'][0].capitalize() + _observables[obs]['desc'][1:] + r" in $" + _process_tex + r"$")
        _obs.tex = r"$" + _observables[obs]['tex'] + r"(" + _process_tex + r")$"
        _obs.add_taxonomy(_process_taxonomy)
        if 'ang_coef' in _observables[obs].keys():
            Prediction(_obs_name, obs_ratio_func(_observables[obs]['func_num'], dGdq2, l, _observables[obs]['ang_coef']))
        else :
            Prediction(_obs_name, obs_ratio_func(_observables[obs]['func_num'], dGdq2, l))
