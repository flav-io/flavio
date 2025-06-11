"""Functions for exclusive $B_s\to V\ell^+\ell^-$ decays, taking into account
the finite life-time difference between the $B_s$ mass eigenstates,
see arXiv:1502.05509."""

import flavio
from . import observables
from flavio.classes import Observable, Prediction
import cmath
import warnings
from .. import angular

### Definite time integrals of the different parts composing the time dependence
def definite_int_cosh(y, Gamma, tmin, tmax):
    den =  (y**2 - 1) # Gamma *
    num_tmax = cmath.exp(-tmax*Gamma) * (y* cmath.sinh(tmax*Gamma*y) + cmath.cosh(tmax*Gamma*y) )
    num_tmin = cmath.exp(-tmin*Gamma) * (y* cmath.sinh(tmin*Gamma*y) + cmath.cosh(tmin*Gamma*y) )
    return (num_tmax - num_tmin)/den

def definite_int_sinh(y, Gamma, tmin, tmax):
    den =  (y**2 - 1) # Gamma *
    num_tmax = cmath.exp(-tmax*Gamma) * (cmath.sinh(tmax*Gamma*y) + y * cmath.cosh(tmax*Gamma*y) )
    num_tmin = cmath.exp(-tmin*Gamma) * (cmath.sinh(tmin*Gamma*y) + y * cmath.cosh(tmin*Gamma*y) )
    return (num_tmax - num_tmin)/den

def definite_int_cos(x, Gamma, tmin, tmax):
    den =  (x**2 + 1) # Gamma *
    num_tmin = cmath.exp(-tmin*Gamma) * (cmath.cos(tmin*Gamma*x) - x * cmath.sin(tmin*Gamma*x) )
    num_tmax = cmath.exp(-tmax*Gamma) * (cmath.cos(tmax*Gamma*x) - x * cmath.sin(tmax*Gamma*x) )
    return (num_tmin - num_tmax)/den

def definite_int_sin(x, Gamma, tmin, tmax):
    den =  (x**2 + 1) # Gamma *
    num_tmin = cmath.exp(-tmin*Gamma) * (cmath.sin(tmin*Gamma*x) + x * cmath.cos(tmin*Gamma*x) )
    num_tmax = cmath.exp(-tmax*Gamma) * (cmath.sin(tmax*Gamma*x) + x * cmath.cos(tmax*Gamma*x) )
    return (num_tmin - num_tmax)/den
    
def bsvll_obs(function, q2, wc_obj, par, B, V, lep):
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mV = par['m_'+V]
    y = par['DeltaGamma/Gamma_'+B]/2.
    x = par['DeltaM/Gamma_'+B]
    gamma = 0.6597 # only for Bs TODO: remove
    if q2 < 4*ml**2 or q2 > (mB-mV)**2:
        return 0
    
    scale = flavio.config['renormalization scale']['bvll']
    mb = flavio.physics.running.running.get_mb(par, scale)
    ff = flavio.physics.bdecays.bvll.amplitudes.get_ff(q2, par, B, V)
    h = flavio.physics.bdecays.bvll.amplitudes.helicity_amps(q2, ff, wc_obj, par, B, V, lep)
    h_bar = flavio.physics.bdecays.bvll.amplitudes.helicity_amps_bar(q2, ff, wc_obj, par, B, V, lep)
    J = flavio.physics.bdecays.angular.angularcoeffs_general_v(h, q2, mB, mV, mb, 0, ml, ml)
    J_bar = flavio.physics.bdecays.angular.angularcoeffs_general_v(h_bar, q2, mB, mV, mb, 0, ml, ml)
    h_tilde = h_bar.copy()
    h_tilde[('pl', 'V')] = h_bar[('mi', 'V')]
    h_tilde[('pl', 'A')] = h_bar[('mi', 'A')]
    h_tilde[('mi', 'V')] = h_bar[('pl', 'V')]
    h_tilde[('mi', 'A')] = h_bar[('pl', 'A')]
    h_tilde['S'] = -h_bar['S']
    q_over_p = flavio.physics.mesonmixing.observables.q_over_p(wc_obj, par, B)
    phi = cmath.phase(-q_over_p) # the phase of -q/p
    J_h = flavio.physics.bdecays.angular.angularcoeffs_h_v(phi, h, h_tilde, q2, mB, mV, mb, 0, ml, ml)
    J_s = flavio.physics.bdecays.angular.angularcoeffs_s_v(phi, h, h_tilde, q2, mB, mV, mb, 0, ml, ml)
    return function(y, x, gamma, J, J_bar, J_h, J_s)

def bsvll_obs_int_t(function, tmin, tmax, q2, wc_obj, par, B, V, lep):
    """
    As above but allows to choose a range for tmin-tmax
    """
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mV = par['m_'+V]
    y = par['DeltaGamma/Gamma_'+B]/2.
    x = par['DeltaM/Gamma_'+B]
    gamma = 0.6597 # only for Bs TODO: remove
    if q2 < 4*ml**2 or q2 > (mB-mV)**2:
        return 0
    
    scale = flavio.config['renormalization scale']['bvll']
    mb = flavio.physics.running.running.get_mb(par, scale)
    ff = flavio.physics.bdecays.bvll.amplitudes.get_ff(q2, par, B, V)
    h = flavio.physics.bdecays.bvll.amplitudes.helicity_amps(q2, ff, wc_obj, par, B, V, lep)
    h_bar = flavio.physics.bdecays.bvll.amplitudes.helicity_amps_bar(q2, ff, wc_obj, par, B, V, lep)
    J = flavio.physics.bdecays.angular.angularcoeffs_general_v(h, q2, mB, mV, mb, 0, ml, ml)
    J_bar = flavio.physics.bdecays.angular.angularcoeffs_general_v(h_bar, q2, mB, mV, mb, 0, ml, ml)
    h_tilde = h_bar.copy()
    h_tilde[('pl', 'V')] = h_bar[('mi', 'V')]
    h_tilde[('pl', 'A')] = h_bar[('mi', 'A')]
    h_tilde[('mi', 'V')] = h_bar[('pl', 'V')]
    h_tilde[('mi', 'A')] = h_bar[('pl', 'A')]
    h_tilde['S'] = -h_bar['S']
    q_over_p = flavio.physics.mesonmixing.observables.q_over_p(wc_obj, par, B)
    phi = cmath.phase(-q_over_p) # the phase of -q/p
    J_h = flavio.physics.bdecays.angular.angularcoeffs_h_v(phi, h, h_tilde, q2, mB, mV, mb, 0, ml, ml)
    J_s = flavio.physics.bdecays.angular.angularcoeffs_s_v(phi, h, h_tilde, q2, mB, mV, mb, 0, ml, ml)
    return function(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s)

def S_original_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    # (42) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    return 1/(1-y**2) * (J[i] + J_bar[i]) - y/(1-y**2) * J_h[i]

def S_original_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -S_original_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)
    return S_original_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)


def S_original_experiment_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    r"""CP-averaged angular observable $S_i$ in the LHCb convention.

    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return S_original_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)/SA_den_Bs(y, x, gamma, J, J_bar, J_h, J_s)

def S_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    # (42) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    if i in [5, '6s', '6c', 8, 9]:
        return 1/(1+x**2) * ((J[i] + J_bar[i]) - x * J_s[i])       
    else:
        return 1/(1-y**2) * ((J[i] + J_bar[i]) - y * J_h[i])

def A_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    # (42) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    if i in [5, '6s', '6c', 8, 9]:
        return 1/(1-y**2) * ((J[i] - J_bar[i]) - y * J_h[i])
    else:
        return 1/(1+x**2) * ((J[i] - J_bar[i]) - x * J_s[i])
    
def S_theory_num_int_t_Bs(tmin, tmax, y, x, Gamma, J, J_bar, J_h, J_s, i):
    # (42) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    if i in [5, '6s', '6c', 8, 9]:
        return ((J[i] + J_bar[i]) * definite_int_cos(x, Gamma, tmin, tmax) 
                - J_s[i] * definite_int_sin(x, Gamma, tmin, tmax))       
    else:
        return ((J[i] + J_bar[i]) * definite_int_cosh(y, Gamma, tmin, tmax) 
                - J_h[i] * definite_int_sinh(y, Gamma, tmin, tmax))

def A_theory_num_int_t_Bs(tmin, tmax, y, x, Gamma, J, J_bar, J_h, J_s, i):
    # (42) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    if i in [5, '6s', '6c', 8, 9]:
        return ((J[i] - J_bar[i]) * definite_int_cosh(y, Gamma, tmin, tmax) 
                - J_h[i] * definite_int_sinh(y, Gamma, tmin, tmax)
                )
    else:
        return ((J[i] - J_bar[i]) * definite_int_cos(x, Gamma, tmin, tmax) 
                - J_s[i] * definite_int_sin(x, Gamma, tmin, tmax)
                )

def K_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    # (42) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    return 1/(1-y**2) * (J[i] + J_bar[i])

def W_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    # (42) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    return 1/(1-y**2) * (J[i] - J_bar[i])

def H_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    # (42) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    return 1/(1-y**2) * J_h[i]

def Z_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    # (42) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    return 1/(1-y**2) * J_s[i]

def S_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -S_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)
    return S_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)

def A_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -A_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)
    return A_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)

def S_experiment_num_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -S_theory_num_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s, i)
    return S_theory_num_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s, i)

def A_experiment_num_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -A_theory_num_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s, i)
    return A_theory_num_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s, i)

def K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -K_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)
    return K_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)

def W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -W_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)
    return W_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)

def H_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -H_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)
    return H_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)

def Z_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -Z_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)
    return Z_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)

def S_experiment_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    r"""CP-averaged angular observable $S_i$ in the LHCb convention.
    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return S_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)/SA_den_Bs(y, x, gamma, J, J_bar, J_h, J_s)

def A_experiment_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    r"""Angular CP asymmetry $A_i$ in the LHCb convention.
    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return A_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)/SA_den_Bs(y, x, gamma, J, J_bar, J_h, J_s)

def S_experiment_int_t_Bs(tmix, tmax, y, x, gamma, J, J_bar, J_h, J_s, i):
    r"""CP-averaged angular observable $S_i$ in the LHCb convention.
    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return S_experiment_num_int_t_Bs(tmix, tmax, y, x, gamma, J, J_bar, J_h, J_s, i)/SA_den_int_t_Bs(tmix, tmax, y, x, gamma, J, J_bar, J_h, J_s)

def A_experiment_int_t_Bs(tmix, tmax, y, x, gamma, J, J_bar, J_h, J_s, i):
    r"""Angular CP asymmetry $A_i$ in the LHCb convention.
    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return A_experiment_num_int_t_Bs(tmix, tmax, y, x, gamma, J, J_bar, J_h, J_s, i)/SA_den_int_t_Bs(tmix, tmax, y, x, gamma, J, J_bar, J_h, J_s)

def K_experiment_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    r"""CP-averaged angular observable $K_i$ from the time-dependent differential decay rate.
    Observables related to the terms depending on \cosh(y\Gamma t)

    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)/SA_den_Bs(y, x, gamma, J, J_bar, J_h, J_s)

def W_experiment_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    r"""Angular CP asymmetry $W_i$ from the time-dependent differential decay rate.
    Observables related to the terms depending on \cosh(y\Gamma t)

    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)/SA_den_Bs(y, x, gamma, J, J_bar, J_h, J_s)

def H_experiment_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    r"""CP-averaged angular observable $H_i$ from the time-dependent differential decay rate.
    Observables related to the terms depending on \sinh(y\Gamma t)

    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return H_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)/SA_den_Bs(y, x, gamma, J, J_bar, J_h, J_s)

def Z_experiment_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    r"""CP-averaged angular observable $Z_i$ from the time-dependent differential decay rate.
    Observables related to the terms depending on \sinh(y\Gamma t)

    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return Z_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)/SA_den_Bs(y, x, gamma, J, J_bar, J_h, J_s)

def dGdq2_interference_Bs(y, x, gamma, J, J_bar, J_h, J_s):
    # TODO: add correct citation
    flavio.citations.register("Descotes-Genon:2015hea")
    return (- y/(1-y**2) * observables.dGdq2(J_h))/2.
    
def dGdq2_ave_Bs(y, x, gamma, J, J_bar, J_h, J_s):
    # (48) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    return (1/(1-y**2) * (observables.dGdq2(J) + observables.dGdq2(J_bar))
            - y/(1-y**2) * observables.dGdq2(J_h))/2.
    
def dGdq2_ave_Bs_part1(y, x, gamma, J, J_bar, J_h, J_s):
    # (48) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    return (1/(1-y**2) * (observables.dGdq2(J) + observables.dGdq2(J_bar)))/2.

def dGdq2_ave_Bs_part2(y, x, gamma, J, J_bar, J_h, J_s):
    # (48) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    return (- y/(1-y**2) * observables.dGdq2(J_h))/2.
    
def dGdq2_ave_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s):
    # (48) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    return ((observables.dGdq2(J) + observables.dGdq2(J_bar)) * definite_int_cosh(y, gamma, tmin, tmax)
            - observables.dGdq2(J_h) * definite_int_sinh(y, gamma, tmin, tmax))/2.

# denominator of S_i and A_i observables
def SA_den_Bs(y, x, gamma, J, J_bar, J_h, J_s):
    return 2*dGdq2_ave_Bs(y, x, gamma, J, J_bar, J_h, J_s)

def SA_den_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s):
    return 2*dGdq2_ave_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s)

def FL_Bs(y, x, gamma, J, J_bar, J_h, J_s):
    r"""Longitudinal polarization fraction $F_L$"""
    return FL_num_Bs(y, x, gamma, J, J_bar, J_h, J_s)/SA_den_Bs(y, x, gamma, J, J_bar, J_h, J_s)

def FL_num_Bs(y, x, gamma, J, J_bar, J_h, J_s):
    return -S_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '2c')

def FL_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s):
    r"""Longitudinal polarization fraction $F_L$"""
    return FL_num_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s)/SA_den_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s)

def FL_num_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s):
    return -S_theory_num_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s, '2c')

def bsvll_dbrdq2(q2, wc_obj, par, B, V, lep):
    tauB = par['tau_'+B]
    return tauB * bsvll_obs(dGdq2_ave_Bs, q2, wc_obj, par, B, V, lep)

def bsvll_obs_int(function, q2min, q2max, wc_obj, par, B, V, lep, epsrel=0.005):
    def obs(q2):
        return bsvll_obs(function, q2, wc_obj, par, B, V, lep)
    return flavio.physics.bdecays.bvll.observables.nintegrate_pole(obs, q2min, q2max, epsrel=epsrel)

def bsvll_dbrdq2_int(q2min, q2max, wc_obj, par, B, V, lep, epsrel=0.005):
    def obs(q2):
        return bsvll_dbrdq2(q2, wc_obj, par, B, V, lep)
    return flavio.physics.bdecays.bvll.observables.nintegrate_pole(obs, q2min, q2max, epsrel=epsrel)/(q2max-q2min)

# Functions returning functions needed for Prediction instances

def bsvll_dbrdq2_int_func(B, V, lep):
    def fct(wc_obj, par, q2min, q2max):
        return bsvll_dbrdq2_int(q2min, q2max, wc_obj, par, B, V, lep)
    return fct

def bsvll_dbrdq2_func(B, V, lep):
    def fct(wc_obj, par, q2):
        return bsvll_dbrdq2(q2, wc_obj, par, B, V, lep)
    return fct

def bsvll_obs_int_ratio_func(func_num, func_den, B, V, lep):
    def fct(wc_obj, par, q2min, q2max):
        num = bsvll_obs_int(func_num, q2min, q2max, wc_obj, par, B, V, lep)
        if num == 0:
            return 0
        denom = bsvll_obs_int(func_den, q2min, q2max, wc_obj, par, B, V, lep)
        return num/denom
    return fct

def bsvll_obs_int_ratio_leptonflavour(func, B, V, l1, l2):
    def fct(wc_obj, par, q2min, q2max):
        num = bsvll_obs_int(func, q2min, q2max, wc_obj, par, B, V, l1, epsrel=0.0005)
        if num == 0:
            return 0
        denom = bsvll_obs_int(func, q2min, q2max, wc_obj, par, B, V, l2, epsrel=0.0005)
        return num/denom
    return fct

def bsvll_obs_ratio_func(func_num, func_den, B, V, lep):
    def fct(wc_obj, par, q2):
        num = bsvll_obs(func_num, q2, wc_obj, par, B, V, lep)
        if num == 0:
            return 0
        denom = bsvll_obs(func_den, q2, wc_obj, par, B, V, lep)
        return num/denom
    return fct

# function for the LHCb (1804.07167) Bs->K*0 integrated branching ratio
def bsvll_dbrdq2_19_func(B, P, lep):
    def fct(wc_obj, par):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The predictions in the region of narrow charmonium resonances are not meaningful.*")
            warnings.filterwarnings("ignore", message="The QCDF corrections should not be trusted*")

            q2max = 19
            q2min = 0.1
            return (1+par['delta_BsKstarmumu'])*bsvll_dbrdq2_int(q2min, q2max, wc_obj, par, B, P, lep)*(q2max-q2min)
    return fct

class BsVll_int_ratio(observables.BVllObservableBinned):
    """Binned ratio of functions if angular coefficients"""
    def __init__(self, func_num, func_den, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsrel = 0.005
        self.func_num = func_num
        self.func_den = func_den
        
    def j_s(self, q2):
        """Get angular coeffs. Cache and only recompute if necessary."""
        h = self.ha(q2)
        if q2 not in self._j:
            self._j[q2] = angular.angularcoeffs_s_v(h, q2, self.mB, self.mV, self.mb, 0, self.ml, self.ml)
        return self._j[q2]

    def j_h(self, q2):
        """Get CP conjugate angular coeffs. Cache and only recompute if necessary."""
        hbar = self.ha_bar(q2)
        if q2 not in self._j_bar:
            self._j_bar[q2] = angular.angularcoeffs_h_v(hbar, q2, self.mB, self.mV, self.mb, 0, self.ml, self.ml)
        return self._j_bar[q2]
    
    def jfunc(self, function, q2):
        """Return a function of J and Jbar at one value of q2."""
        return function(self.j(q2), self.jbar(q2), self.j_h(q2), self.j_s(q2))
    
    def obs_num(self, q2):
        return self.jfunc(self.func_num, q2)

    def obs_den(self, q2):
        return self.jfunc(self.func_den, q2)

    def __call__(self):
        if self.q2max_allowed <= self.q2min_allowed:
            return 0
        num = flavio.physics.bdecays.bvll.observables.nintegrate_pole(self.obs_num, self.q2min_allowed, self.q2max_allowed, epsrel=self.epsrel)
        if num == 0:
            return 0
        den = flavio.physics.bdecays.bvll.observables.nintegrate_pole(self.obs_den, self.q2min_allowed, self.q2max_allowed, epsrel=self.epsrel)
        return num / den

# Observable and Prediction instances

_tex = {'e': 'e', 'mu': '\mu', 'tau': r'\tau'}
_observables = {
'FL': {'func_num': FL_num_Bs, 'tex': r'\overline{F_L}', 'desc': 'Time-averaged longitudinal polarization fraction'},
'S3': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: S_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 3), 'tex': r'\overline{S_3}', 'desc': 'Time-averaged, CP-averaged angular observable'},
'S4': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: S_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 4), 'tex': r'\overline{S_4}', 'desc': 'Time-averaged, CP-averaged angular observable'},
'S7': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: S_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 7), 'tex': r'\overline{S_7}', 'desc': 'Time-averaged, CP-averaged angular observable'},
}
_hadr = {
'Bs->phi': {'tex': r"B_s\to \phi ", 'B': 'Bs', 'V': 'phi', },
'Bs->K*0': {'tex': r"B_s\to K^* ", 'B': 'Bs', 'V': 'K*0', },
}
for l in ['e', 'mu', 'tau']:
    for M in _hadr.keys():

        _process_tex = _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+r"^-"
        _process_taxonomy = r'Process :: $b$ hadron decays :: FCNC decays :: $B\to V\ell^+\ell^-$ :: $' + _process_tex + r"$"

        for obs in sorted(_observables.keys()):

            # binned angular observables
            _obs_name = "<" + obs + ">("+M+l+l+")"
            _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
            _obs.set_description('Binned ' + _observables[obs]['desc'] + r" in $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$")
            _obs.tex = r"$\langle " + _observables[obs]['tex'] + r"\rangle(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
            _obs.add_taxonomy(_process_taxonomy)
            Prediction(_obs_name, bsvll_obs_int_ratio_func(_observables[obs]['func_num'], SA_den_Bs, _hadr[M]['B'], _hadr[M]['V'], l))

            # differential angular observables
            _obs_name = obs + "("+M+l+l+")"
            _obs = Observable(name=_obs_name, arguments=['q2'])
            _obs.set_description(_observables[obs]['desc'][0].capitalize() + _observables[obs]['desc'][1:] + r" in $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$")
            _obs.tex = r"$" + _observables[obs]['tex'] + r"(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
            _obs.add_taxonomy(_process_taxonomy)
            Prediction(_obs_name, bsvll_obs_ratio_func(_observables[obs]['func_num'], SA_den_Bs, _hadr[M]['B'], _hadr[M]['V'], l))

        # binned branching ratio
        _obs_name = "<dBR/dq2>("+M+l+l+")"
        _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
        _obs.set_description(r"Binned time-integrated differential branching ratio of $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$")
        _obs.tex = r"$\langle \frac{d\overline{\text{BR}}}{dq^2} \rangle(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
        _obs.add_taxonomy(_process_taxonomy)
        Prediction(_obs_name, bsvll_dbrdq2_int_func(_hadr[M]['B'], _hadr[M]['V'], l))

        # differential branching ratio
        _obs_name = "dBR/dq2("+M+l+l+")"
        _obs = Observable(name=_obs_name, arguments=['q2'])
        _obs.set_description(r"Differential time-integrated branching ratio of $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$")
        _obs.tex = r"$\frac{d\overline{\text{BR}}}{dq^2}(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
        _obs.add_taxonomy(_process_taxonomy)
        Prediction(_obs_name, bsvll_dbrdq2_func(_hadr[M]['B'], _hadr[M]['V'], l))

        # for Bs->K*0 mu mu we add a separate observable for the measurement in 1804.07167
        if M == 'Bs->K*0' and l == 'mu':
            _obs_name = "BR_LHCb("+M+l+l+")"
            _obs = Observable(name=_obs_name)
            _obs.set_description(r"Branching ratio of $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$ measured by LHCb in 2018")
            _obs.tex = r"$\overline{\text{BR}}(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
            _obs.add_taxonomy(_process_taxonomy)
            Prediction(_obs_name, bsvll_dbrdq2_19_func(_hadr[M]['B'], _hadr[M]['V'], l))

# Lepton flavour ratios
for l in [('mu','e'), ('tau','mu'),]:
    for M in _hadr.keys():

        # binned ratio of BRs
        _obs_name = "<R"+l[0]+l[1]+">("+M+"ll)"
        _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
        _obs.set_description(r"Ratio of partial branching ratios of $" + _hadr[M]['tex'] +_tex[l[0]]+r"^+ "+_tex[l[0]]+r"^-$" + " and " + r"$" + _hadr[M]['tex'] +_tex[l[1]]+r"^+ "+_tex[l[1]]+"^-$")
        _obs.tex = r"$\langle R_{" + _tex[l[0]] + ' ' + _tex[l[1]] + r"} \rangle(" + _hadr[M]['tex'] + r"\ell^+\ell^-)$"
        for li in l:
            # add taxonomy for both processes (e.g. Bs->Vee and Bs->Vmumu)
            _obs.add_taxonomy(r'Process :: $b$ hadron decays :: FCNC decays :: $B\to V\ell^+\ell^-$ :: $' + _hadr[M]['tex'] +_tex[li]+r"^+"+_tex[li]+r"^-$")
        Prediction(_obs_name, bsvll_obs_int_ratio_leptonflavour(dGdq2_ave_Bs, _hadr[M]['B'], _hadr[M]['V'], *l))
