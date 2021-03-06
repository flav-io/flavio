"""Functions for exclusive $B_s\to V\ell^+\ell^-$ decays, taking into account
the finite life-time difference between the $B_s$ mass eigenstates,
see arXiv:1502.05509."""

import flavio
from . import observables
from flavio.classes import Observable, Prediction
import cmath

def bsvll_obs(function, q2, wc_obj, par, B, V, lep):
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mV = par['m_'+V]
    y = par['DeltaGamma/Gamma_'+B]/2.
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
    return function(y, J, J_bar, J_h)

def S_theory_num_Bs(y, J, J_bar, J_h, i):
    # (42) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    return 1/(1-y**2) * (J[i] + J_bar[i]) - y/(1-y**2) * J_h[i]

def S_experiment_num_Bs(y, J, J_bar, J_h, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -S_theory_num_Bs(y, J, J_bar, J_h, i)
    return S_theory_num_Bs(y, J, J_bar, J_h, i)


def S_experiment_Bs(y, J, J_bar, J_h, i):
    r"""CP-averaged angular observable $S_i$ in the LHCb convention.

    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return S_experiment_num_Bs(y, J, J_bar, J_h, i)/SA_den_Bs(y, J, J_bar, J_h)

def dGdq2_ave_Bs(y, J, J_bar, J_h):
    # (48) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    return (1/(1-y**2) * (observables.dGdq2(J) + observables.dGdq2(J_bar))
            - y/(1-y**2) * observables.dGdq2(J_h))/2.

# denominator of S_i and A_i observables
def SA_den_Bs(y, J, J_bar, J_h):
    return 2*dGdq2_ave_Bs(y, J, J_bar, J_h)

def FL_Bs(y, J, J_bar, J_h):
    r"""Longitudinal polarization fraction $F_L$"""
    return FL_num_Bs(y, J, J_bar, J_h)/SA_den_Bs(y, J, J_bar, J_h)

def FL_num_Bs(y, J, J_bar, J_h):
    return -S_theory_num_Bs(y, J, J_bar, J_h, '2c')


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

# Observable and Prediction instances

_tex = {'e': 'e', 'mu': '\mu', 'tau': r'\tau'}
_observables = {
'FL': {'func_num': FL_num_Bs, 'tex': r'\overline{F_L}', 'desc': 'Time-averaged longitudinal polarization fraction'},
'S3': {'func_num': lambda y, J, J_bar, J_h: S_experiment_num_Bs(y, J, J_bar, J_h, 3), 'tex': r'\overline{S_3}', 'desc': 'Time-averaged, CP-averaged angular observable'},
'S4': {'func_num': lambda y, J, J_bar, J_h: S_experiment_num_Bs(y, J, J_bar, J_h, 4), 'tex': r'\overline{S_4}', 'desc': 'Time-averaged, CP-averaged angular observable'},
'S7': {'func_num': lambda y, J, J_bar, J_h: S_experiment_num_Bs(y, J, J_bar, J_h, 7), 'tex': r'\overline{S_7}', 'desc': 'Time-averaged, CP-averaged angular observable'},
}
_hadr = {
'Bs->phi': {'tex': r"B_s\to \phi ", 'B': 'Bs', 'V': 'phi', },
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
