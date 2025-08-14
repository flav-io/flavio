r""" Functions to compute the transversity amplitudes for $B_s \to V \ell^+ \ell^-$ decays. """
from flavio.physics.bdecays.common import lambda_K
from math import sqrt, pi
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict, get_wceff
from flavio.physics.running import running
from flavio.config import config
from flavio.physics.bdecays.common import meson_quark, meson_ff
from flavio.physics.common import conjugate_par, conjugate_wc
from flavio.physics import ckm
from flavio.classes import AuxiliaryQuantity
import warnings

def get_ff(q2, par, B, V):
    ff_name = meson_ff[(B,V)] + ' form factor'
    return AuxiliaryQuantity[ff_name].prediction(par_dict=par, wc_obj=None, q2=q2)

def prefactor(q2, par, B, V, mB, mV, mL):
    """ prefactor N from https://arxiv.org/pdf/0811.1214 (3.33) """
    GF = par['GF']
    scale = config['renormalization scale']['bvll']
    alphaem = running.get_alpha(par, scale)['alpha_e']
    di_dj = meson_quark[(B,V)]
    xi_t = ckm.xi('t', di_dj)(par)
    lambda_b = lambda_K(mB**2, mV**2, q2)
    betal = sqrt( 1 - 4 * mL**2 / q2 )
    return xi_t * GF * alphaem * ( q2 * lambda_b**0.5 * betal / 3 / 2**10 / pi**5 / mB**3 )**0.5

def transversity_amps_ff(q2, ff, wc_obj, par_dict, B, V, lep, cp_conjugate): 
    """ transversity amplitudes for B->Vll from https://arxiv.org/pdf/0811.1214 """
    par = conjugate_par(par_dict.copy()) if cp_conjugate else par_dict.copy()
    scale = config['renormalization scale']['bvll']
    label = meson_quark[(B,V)] + lep + lep  # e.g. bsmumu, bdtautau
    wc = wctot_dict(wc_obj, label, scale, par)
    if cp_conjugate:
        wc = conjugate_wc(wc)
    wc_eff = get_wceff(q2, wc, par, B, V, lep, scale) 
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mV = par['m_'+V]
    mb = running.get_mb(par, scale)
    N = prefactor(q2, par, B, V, mB, mV, ml)
    return compute_transversity_amps(q2, mB, mV, mb, 0, ml, ml, ff, wc_eff, N)

def compute_transversity_amps(q2, mB, mV, mqh, mql, ml1, ml2, ff, wc, prefactor):
    """
    transversity amplitudes for B -> V ll decays. 
    Taken from https://arxiv.org/pdf/0811.1214 EQ (3.28) and following. 
    prefactor refers to the N from the paper above. 
    N = Vtb Vts* sqrt{GF^2 alpha_em^2 q^2 beta_mu sqrt(lambda_K(mB^2, mV^2, q^2)) / [3 * 2^10 pi^5 mB^3]}

    NB: - light quark mass not used here, but kept for consistency in the function signature with helicity_amps_v. 
        - ml1 and ml2 are always the same lepton mass. It therefore does not matter which one is used for the A_t definition. 
        - transversity amplitudes are also given in https://arxiv.org/pdf/hep-ph/0502060
    """
    def lambda_qsq(q2, mB, mV):
        return ((mB + mV)**2 - q2) * ((mB - mV)**2 - q2)  # (D.3) in https://arxiv.org/pdf/1503.05534
    lambda_b = lambda_K(mB**2, mV**2, q2)
    wc_9 = wc['v']
    wc_9p = wc['vp']
    wc_10 = wc['a']
    wc_10p = wc['ap']
    wc_7, wc_7p = wc['7'], wc['7p']
    wc_p, wc_pp = wc['p'], wc['pp']

    # (D.5) in https://arxiv.org/pdf/1503.05534
    ff['A2'] = -(mB + mV)*(-ff['A1']*mB**3 - ff['A1']*mB**2*mV + ff['A1']*mB*mV**2 + ff['A1']*mB*q2 + ff['A1']*mV**3 + ff['A1']*mV*q2 + 16*ff['A12']*mB*mV**2)/lambda_qsq(q2, mB, mV)
    ff['T3'] = -(mB - mV)*(-ff['T2']*mB**3 - ff['T2']*mB**2*mV - 3*ff['T2']*mB*mV**2 + ff['T2']*mB*q2 - 3*ff['T2']*mV**3 + ff['T2']*mV*q2 + 8*ff['T23']*mB*mV**2)/lambda_qsq(q2, mB, mV)

    # from https://arxiv.org/pdf/0811.1214
    a0_l_term_one = ( (wc_9 - wc_9p) - (wc_10 - wc_10p) ) * ( (mB**2 - mV**2 - q2) * (mB + mV) * ff['A1'] - lambda_b * ff['A2'] / (mB + mV) )  # first summand of (3.30)
    a0_r_term_one = ( (wc_9 - wc_9p) + (wc_10 - wc_10p) ) * ( (mB**2 - mV**2 - q2) * (mB + mV) * ff['A1'] - lambda_b * ff['A2'] / (mB + mV) )  
    a0_term_two = 2 * mqh * (wc_7 - wc_7p) * ( ( mB**2 + 3 * mV**2 - q2 ) * ff['T2'] - lambda_b / (mB**2 - mV**2) * ff['T3'] )  # second summand of (3.30) 

    transversity_amps = {
        'perp_L': sqrt(2 * lambda_b) * ( ( (wc_9 + wc_9p) - (wc_10 + wc_10p)) * ff['V']/(mB + mV) + 2 * mqh / q2 * (wc_7 + wc_7p) * ff['T1'] ),  # (3.28)
        'perp_R': sqrt(2 * lambda_b) * ( ( (wc_9 + wc_9p) + (wc_10 + wc_10p)) * ff['V']/(mB + mV) + 2 * mqh / q2 * (wc_7 + wc_7p) * ff['T1'] ),  # (3.28)
        'para_L': -sqrt(2) * (mB**2 - mV**2) * ( ((wc_9 - wc_9p) - (wc_10 - wc_10p)) * ff['A1']/(mB - mV) + 2 * mqh / q2 * (wc_7 - wc_7p) * ff['T2'] ),  # (3.29)
        'para_R': -sqrt(2) * (mB**2 - mV**2) * ( ((wc_9 - wc_9p) + (wc_10 - wc_10p)) * ff['A1']/(mB - mV) + 2 * mqh / q2 * (wc_7 - wc_7p) * ff['T2'] ),  # (3.29)
        '0_L': - 1 / (2 * mV * sqrt(q2)) * ( a0_l_term_one + a0_term_two ),  # (3.30)
        '0_R': - 1 / (2 * mV * sqrt(q2)) * ( a0_r_term_one + a0_term_two ),  # (3.30)
        't': sqrt(lambda_b / q2) * ff['A0'] * ( 2 * (wc_10 - wc_10p) + q2 / ml1 * (wc_p - wc_pp) ),  # (3.31)  
        'S': - 2 * sqrt(lambda_b) * (wc['s'] - wc['sp']) * ff['A0'],  # (3.32)
    }

    return {k: prefactor * v for k, v in transversity_amps.items()}

def _Re(z):
    return z.real

def _Im(z):
    return z.imag

def _Co(z):
    return complex(z).conjugate()

def angularcoeffs_general_transversity(A, q2, ml):
    """
    Returns the angular coefficients from the transversity amplitudes, 
    compare e.g. https://arxiv.org/pdf/1502.05509 EQ 9.
    The transversity amplitudes are stored in the dictionary A with the keys:
     - para_L,R, 
     - perp_L,R, 
     - 0_L,R, 
     - t, 
     - S

    NB: phi is not used in this function, but is included for consistency with the other functions.
    """
    def _CAS(x):  # complex absolute square
        return x * _Co(x)

    beta_l = sqrt(1 - 4 * ml**2 / q2)

    J = {
        '1s': (2 + beta_l**2) / 4 * ( _CAS(A['para_L']) + _CAS(A['para_R']) + _CAS(A['perp_L']) + _CAS(A['perp_R']) ) 
              + 4 * ml**2 / q2 * _Re( A['perp_L'] * _Co(A['perp_R']) + A['para_L'] * _Co(A['para_R']) ),
        '1c': (_CAS(A['0_L']) + _CAS(A['0_R'])) + 4 * ml**2 / q2 * ( _CAS(A['t']) + 2 * _Re( A['0_L'] * _Co(A['0_R']) ) ) + beta_l**2 * _CAS(A['S']),
        '2s': beta_l**2 / 4 * ( _CAS(A['para_L']) + _CAS(A['para_R']) + _CAS(A['perp_L']) + _CAS(A['perp_R']) ),
        '2c': - 1 * beta_l**2 * (_CAS(A['0_L']) + _CAS(A['0_R'])),
        3: beta_l**2 / 2 * ( _CAS(A['perp_L']) - _CAS(A['para_L']) + _CAS(A['perp_R']) - _CAS(A['para_R']) ),
        4: beta_l**2 / sqrt(2) * ( _Re( A['0_L'] * _Co(A['para_L']) + A['0_R'] * _Co(A['para_R']) ) ),
        5: sqrt(2) * beta_l * ( _Re( A['0_L'] * _Co(A['perp_L']) - A['0_R'] * _Co(A['perp_R']) ) - ml / sqrt(q2) * _Re( A['para_L'] * _Co(A['S']) + A['para_R'] * _Co(A['S']) ) ),  # differs between https://arxiv.org/pdf/0811.1214 (this version) and https://arxiv.org/pdf/1502.05509
        '6s': 2 * beta_l * ( _Re( A['para_L'] * _Co(A['perp_L']) - A['para_R'] * _Co(A['perp_R']) ) ),
        '6c': 4 * beta_l * ml / sqrt(q2) * ( _Re( A['0_L'] * _Co(A['S']) + A['0_R'] * _Co(A['S']) ) ),  # differs between https://arxiv.org/pdf/0811.1214 (this version) and https://arxiv.org/pdf/1502.05509
        7: sqrt(2) * beta_l * ( _Im( A['0_L'] * _Co(A['para_L']) - A['0_R'] * _Co(A['para_R']) ) + ml / sqrt(q2) * _Im( A['perp_L'] * _Co(A['S']) - A['perp_R'] * _Co(A['S']) ) ),  # differs between https://arxiv.org/pdf/0811.1214 (this version) and https://arxiv.org/pdf/1502.05509
        8: beta_l**2 / sqrt(2) * ( _Im( A['0_L'] * _Co(A['perp_L']) + A['0_R'] * _Co(A['perp_R']) ) ),
        9: beta_l**2  * ( _Im( _Co(A['para_L']) * A['perp_L'] + _Co(A['para_R']) * A['perp_R'] ) ),
        # 7, 8, 9 differ between https://arxiv.org/pdf/1502.05509 (this version) https://arxiv.org/pdf/0811.1214 (suspect parenthesis typo with L->R writing)
    }
    return J

def angularcoeffs_h_transversity(A, Atilde, q2, ml, qp) -> dict[str | int, float]: 
    """ 
    Returns the angular coefficients h_i from the transversity amplitudes. 
    Compare e.g. https://arxiv.org/pdf/1502.05509 Appendix C, EQ 117 and following. 
    """
    qp = -qp
    beta_l = sqrt(1 - 4 * ml**2 / q2)
    beta_l2 = 1 - 4 * ml**2 / q2

    AtL_ALs_perp = Atilde['perp_L'] * _Co(A['perp_L'])
    AtR_ARs_perp = Atilde['perp_R'] * _Co(A['perp_R'])
    AtL_ALs_para = Atilde['para_L'] * _Co(A['para_L'])
    AtR_ARs_para = Atilde['para_R'] * _Co(A['para_R'])
    AtL_ALs_0 = Atilde['0_L'] * _Co(A['0_L'])
    AtR_ARs_0 = Atilde['0_R'] * _Co(A['0_R'])
    At_As_t = Atilde['t'] * _Co(A['t'])
    At_As_S = Atilde['S'] * _Co(A['S'])

    AtL_ARs_perp = Atilde['perp_L'] * _Co(A['perp_R'])
    AtR_ALs_perp = Atilde['perp_R'] * _Co(A['perp_L'])
    AtL_ARs_para = Atilde['para_L'] * _Co(A['para_R'])
    AtR_ALs_para = Atilde['para_R'] * _Co(A['para_L'])

    AtL_ARs_0 = Atilde['0_L'] * _Co(A['0_R'])
    AtR_ALs_0 = Atilde['0_R'] * _Co(A['0_L'])

    h = {
        '1s': (2 + beta_l2) / 2 * _Re( qp * ( AtL_ALs_perp + AtL_ALs_para + AtR_ARs_perp + AtR_ARs_para ) )
              + 4 * ml**2 / q2 * _Re( qp * ( AtL_ARs_perp + AtL_ARs_para ) + _Co(qp) * ( _Co( AtR_ALs_perp ) * _Co(AtR_ALs_para) ) ),  # (117)
        '1c': 2 * _Re( qp * ( AtL_ALs_0 + AtR_ARs_0 ) ) 
              + 8 * ml**2 / q2 * (_Re( qp * At_As_t ) + _Re( qp * AtL_ARs_0 + _Co(qp) * _Co(AtR_ALs_0) ))
              + 2 * beta_l2 * _Re( qp * At_As_S ),  # (118)
        '2s': beta_l2 / 2 * _Re( qp * ( AtL_ALs_perp + AtL_ALs_para + AtR_ARs_perp + AtR_ARs_para ) ),  # (119) (= 1s for massless leptons)
        '2c': -2 * beta_l2 * _Re( qp * ( AtL_ALs_0 + AtR_ARs_0 ) ),  # (120) (= -1c for massless leptons)
        3: beta_l2 * _Re( qp * ( AtL_ALs_perp - AtL_ALs_para + AtR_ARs_perp - AtR_ARs_para ) ),  # (121)
        4: beta_l2 / sqrt(2) * _Re( qp * (Atilde['0_L'] * _Co(A['para_L']) + Atilde['0_R'] * _Co(A['para_R'])) 
                                   + _Co(qp) * ( A['0_L'] * _Co(Atilde['para_L']) + A['0_R'] * _Co(Atilde['para_R']) ) ),  # (122)
        5: sqrt(2) * beta_l * (_Re( qp * ( Atilde['0_L'] * _Co(A['perp_L']) - Atilde['0_R'] * _Co(A['perp_R']) ) 
                                   + _Co(qp) * (A['0_L'] * _Co(Atilde['perp_L']) - A['0_R'] * _Co(Atilde['perp_R'])) ) 
                                 - ml / sqrt(q2) * _Re( qp * ( Atilde['para_L'] * _Co(A['S']) + Atilde['para_R'] * _Co(A['S']) ) 
                                                       + _Co(qp) * ( A['para_L'] * _Co(Atilde['S']) + A['para_R'] * _Co(Atilde['S']) ) ) ),  # (123)
        '6s': 2 * beta_l * _Re( 
            qp * ( Atilde['para_L'] * _Co(A['perp_L']) - Atilde['para_R'] * _Co(A['perp_R']) ) 
            + _Co(qp) * ( A['para_L'] * _Co(Atilde['perp_L']) - A['para_R'] * _Co(Atilde['perp_R']) ) 
        ),  # (124)
        '6c': 4 * beta_l * ml / sqrt(q2) * _Re( qp * ( Atilde['0_L'] * _Co(A['S'] + Atilde['0_R'] * _Co(A['S'])) ) 
                                               + _Co(qp) * ( A['0_L'] * _Co(Atilde['S']) + A['0_R'] * _Co(Atilde['S']) ) ),  # (125)
        7: sqrt(2) * beta_l * ( 
            _Im( qp * ( Atilde['0_L'] * _Co(A['para_L']) - Atilde['0_R'] * _Co(A['para_R']) ) 
                + _Co(qp) * ( A['0_L'] * _Co(Atilde['para_L']) - A['0_R'] * _Co(Atilde['para_R']) ) ) 
            + ml / sqrt(q2) * _Im( qp * ( Atilde['perp_L'] * _Co(A['S']) + Atilde['perp_R'] * _Co(A['S']) ) 
                                  + _Co(qp) * ( A['perp_L'] * _Co(Atilde['S']) + A['perp_R'] * _Co(Atilde['S']) ) ) 
        ),  # (126)
        8: beta_l2 / sqrt(2) * _Im( qp * ( Atilde['0_L'] * _Co(A['perp_L']) + Atilde['0_R'] * _Co(A['perp_R']) ) 
                                   + _Co(qp) * ( A['0_L'] * _Co(Atilde['perp_L']) + A['0_R'] * _Co(Atilde['perp_R']) ) ),  # (127)
        9: -beta_l2 * _Im( qp * ( Atilde['para_L'] * _Co(A['perp_L']) + Atilde['para_R'] * _Co(A['perp_R']) ) 
                          + _Co(qp) * ( A['para_L'] * _Co(Atilde['perp_L']) + A['para_R'] * _Co(Atilde['perp_R']) ) ),  # (128)
    }
    return h

def angularcoeffs_s_transversity(A, Atilde, q2, ml, qp) -> dict[str | int, float]: 
    """ 
    Returns the angular coefficients h_i from the transversity amplitudes. 
    Compare e.g. https://arxiv.org/pdf/1502.05509 Appendix C, EQ 105 and following. 
    """
    qp = -qp
    beta_l = sqrt(1 - 4 * ml**2 / q2)
    beta_l2 = 1 - 4 * ml**2 / q2

    s = {
        '1s': (2 + beta_l2) / 2 * _Im( qp * ( Atilde['perp_L'] * _Co(A['perp_L']) + Atilde['para_L'] * _Co(A['para_L']) 
                                             + Atilde['perp_R'] * _Co(A['perp_R']) + Atilde['para_R'] * _Co(A['para_R']) ) ) 
              + 4 * ml**2 / q2 * _Im( qp * ( Atilde['perp_L'] * _Co(A['perp_R']) + Atilde['para_L'] * _Co(A['para_R']) ) 
                                     - _Co(qp) * ( A['perp_L'] * _Co(Atilde['perp_R']) + A['para_L'] * _Co(Atilde['para_R']) ) ),  # (105)
        '1c': 2 * _Im( qp * ( Atilde['0_L'] * _Co(A['0_L']) + Atilde['0_R'] * _Co(A['0_R']) ) ) 
              + 8 * ml**2 / q2 * ( _Im( qp * Atilde['t'] * _Co(A['t']) ) + _Im( qp * Atilde['0_L'] * _Co(A['0_R']) 
                                                                               - _Co(qp) * A['0_L'] * _Co(Atilde['0_R']) ) )
              + 2 * beta_l2 * _Im( qp * Atilde['S'] * _Co(A['S']) ),  # (106)
        '2s': beta_l2 / 2 * _Im( qp * ( Atilde['perp_L'] * _Co(A['perp_L']) + Atilde['para_L'] * _Co(A['para_L']) 
                                       + Atilde['perp_R'] * _Co(A['perp_R']) + Atilde['para_R'] * _Co(A['para_R']) ) ),  # (107)
        '2c': -2 * beta_l2 * _Im( qp * ( Atilde['0_L'] * _Co(A['0_L']) + Atilde['0_R'] * _Co(A['0_R']) ) ),  # (108)
        3: beta_l2 * _Im( qp * ( Atilde['perp_L'] * _Co(A['perp_L']) - Atilde['para_L'] * _Co(A['para_L']) 
                                + Atilde['perp_R'] * _Co(A['perp_R']) - Atilde['para_R'] * _Co(A['para_R']) ) ),  # (109)
        4: beta_l2 / sqrt(2) * _Im( qp * ( Atilde['0_L'] * _Co(A['para_L']) + Atilde['0_R'] * _Co(A['para_R']) ) 
                                   - _Co(qp) * ( A['0_L'] * _Co(Atilde['para_L']) + A['0_R'] * _Co(Atilde['para_R']) ) ),  # (110)
        5: sqrt(2) * beta_l * ( _Im( qp * ( Atilde['0_L'] * _Co(A['perp_L']) - Atilde['0_R'] * _Co(A['perp_R']) ) 
                                    - _Co(qp) * ( A['0_L'] * _Co(Atilde['perp_L']) - A['0_R'] * _Co(Atilde['perp_R']) ) )
                               - ml / sqrt(2) * _Im( qp * ( Atilde['para_L'] * _Co(A['S']) + Atilde['para_R'] * _Co(A['S']) ) 
                                                    - _Co(qp) * ( A['para_L'] * _Co(Atilde['S']) + A['para_R'] * _Co(Atilde['S']) ) ) ),  # (111)
        '6s': 2 * beta_l * _Im( qp * ( Atilde['para_L'] * _Co(A['perp_L']) - Atilde['para_R'] * _Co(A['perp_R']) ) 
        - _Co(qp) * ( A['para_L'] * _Co(Atilde['perp_L']) - A['para_R'] * _Co(Atilde['perp_R']) ) ),  # (112)
        '6c': 4 * beta_l * ml / sqrt(q2) * _Im( qp * ( Atilde['0_L'] * _Co(A['S']) + Atilde['0_R'] * _Co(A['S']) ) 
        - _Co(qp) * ( A['0_L'] * _Co(Atilde['S']) + A['0_R'] * _Co(Atilde['S']) ) ),  # (113)
        7: -sqrt(2) * beta_l * ( _Re( qp * ( Atilde['0_L'] * _Co(A['para_L']) - Atilde['0_R'] * _Co(A['para_R']) ) 
                                     - _Co(qp) * ( A['0_L'] * _Co(Atilde['para_L']) - A['0_R'] * _Co(Atilde['para_R']) ) ) 
                                + ml / sqrt(q2) * _Re( qp * ( Atilde['perp_L'] * _Co(A['S']) + Atilde['perp_R'] * _Co(A['S']) ) 
                                                      - _Co(qp) * ( A['perp_L'] * _Co(Atilde['S']) + A['perp_R'] * _Co(Atilde['S']) ) ) ),  # (114)
        8: -beta_l2 / sqrt(2) * _Re( qp * ( Atilde['0_L'] * _Co(A['perp_L']) + Atilde['0_R'] * _Co(A['perp_R']) ) 
                                    - _Co(qp) * ( A['0_L'] * _Co(Atilde['perp_L']) + A['0_R'] * _Co(Atilde['perp_R']) ) ),  # (115)
        9: beta_l2 * _Re( qp * ( Atilde['para_L'] * _Co(A['perp_L']) + Atilde['para_R'] * _Co(A['perp_R']) ) 
                         - _Co(qp) * ( A['para_L'] * _Co(Atilde['perp_L']) + A['para_R'] * _Co(Atilde['perp_R']) ) ),  # (116)
    }    
    return s

def transversity_amps(q2, ff, wc_obj, par, B, V, lep):
    """ 
    transversity amplitudes (CP conjugated) for Bs->Vll.
    NOTE: these are computed without the subleading hadronic contributions and QCDF corrections!
    """ 
    if q2 >= 8.7 and q2 < 14:
        warnings.warn("The predictions in the region of narrow charmonium resonances are not meaningful")
    return transversity_amps_ff(q2, ff, wc_obj, par, B, V, lep, cp_conjugate=False)

def transversity_amps_bar(q2, ff, wc_obj, par, B, V, lep): 
    """ 
    transversity amplitudes (CP conjugated) for Bs->Vll.
    NOTE: these are computed without the subleading hadronic contributions and QCDF corrections!
    """ 
    if q2 >= 8.7 and q2 < 14:
        warnings.warn("The predictions in the region of narrow charmonium resonances are not meaningful")
    return transversity_amps_ff(q2, ff, wc_obj, par, B, V, lep, cp_conjugate=True)