r"""Functions for $\Lambda_b \to \Lambda(1520)(\to NK) \ell^+ \ell⁻$ decays as in arXiv:1903.00448."""

import flavio
from match import sqrt, pi
from flavio.physics.bdecays.common import lambda_k, beta_l, meson_quark, meson_ff
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
    # br is br(l(1520)->kp)

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
    l['2ss'] = br/8*( 2*abs(ta['Apara0','L'])**2 + abs(ta['Apara1','L'])**2
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
    

# def get_ff(q2, par) -> form factors from auxiliaryquantity computed in formfactor-directory

def prefactor(q2, par, scale):
    #calculate prefactor N
    xi_t = flavio.physics.ckm.xi('t','bs')(par)
    alphaem = flavio.physics.running.running.get_alpha(par, scale)['alpha_e']
    mLb = par['m_Lambdab']
    mL = par['m_Lambda(1520)']
    la_K = flavio.physics.bdecays.common.lambda_K(mlb**2, ml**2, q2)
    return par['GF'] * xi_t * alphaem * sqrt(q2) * la_K**(1/4.) / sqrt(3 * 2 * mLb**3 * pi**5) / 32


def get_transversity_amps_ff(q2, wc_obj, par_dict, lep, cp_conjugate):
    par = par_dict.copy()
    if cp_conjugate:
        par = conjugate_par(par)
    scale = flavio.config['renormalization scale']['lambdab']
    mlb = par['m_lambdab']
    ml = par['m_lambda']
    mb = flavio.physics.running.running.get_mb(par, scale)
    # !!! get_ff !!!
    ff = get_ff(q2, par)
    wc = flavio.physics.bdecays.wilsoncoefficients.wctot_dict(wc_obj, 'BS' + lep + lep, scale, par)
    wc_eff = flavio.physics.bdecays.wilsoncoefficients.get_wceff(q2, wc, par, 'Lambdab', 'Lambda(1520)', lep, scale)
    ha = helicity_amps(q2, mLb, mL, ff)
    N = prefactor(q2, par, scale)
    ta_ff = transversity_amps(ha, q2, mLb, mL, mb, 0, wc_eff, N)
    return ta_ff


# def get_transversity_amps_ff -> get helicity_amps+prefactor->transversity_amps

# defget_subleading -> subleading hadronic contrubtions at low q2

# def get_transversity_amps -> get_transversity_amps_ff + get_subleading

# def get_obs -> L's

# def dGdq2(K), FL_num(K), AFBl_num(K), AFBh_num(K), AFBlh_num(K), dbrdq2, dbrdq2_int, obs_int, ...
