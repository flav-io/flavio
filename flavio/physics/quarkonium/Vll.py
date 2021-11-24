r"""$B_c$ lifetime"""

import flavio

def BrJpsi(wc_obj, par):
    r"""Branching ratio for the lepton-flavour violating leptonic decay J/psi-> l l' based on XXXX.XXXXX"""
    Gamma_SM = 1 / par['tau_Bc_SM']
    Gamma_exp = 1 / par['tau_Bc']
    Gamma_NP = 0
    wc_sm = flavio.WilsonCoefficients()
    for l in ['e', 'mu', 'tau']:
        _br_SM = flavio.Observable['BR(Bc->{}nu)'.format(l)].prediction_par(par, wc_sm)
        _br_NP = flavio.Observable['BR(Bc->{}nu)'.format(l)].prediction_par(par, wc_obj)
        Gamma_NP += (_br_NP - _br_SM) * Gamma_exp
    return 1 / (Gamma_SM + Gamma_NP)

# Observable and Prediction instances
_hadr_lfv = {
'J/psi': {'tex': r"J/\psi\to", 'V': 'J/psi', },
}
_tex_lfv = {'emu': r'e^+\mu^-', 'mue': r'\mu^+e^-',
    'taue': r'\tau^+e^-', 'etau': r'e^+\tau^-',
    'taumu': r'\tau^+\mu^-', 'mutau': r'\mu^+\tau^-',
    'emu,mue': r'e^\pm\mu^\mp', 'etau,taue': r'e^\pm\tau^\mp',
    'mutau,taumu': r'\mu^\pm\tau^\mp'}


def _define_obs_V_ll(M, ll):
    _process_tex = _hadr_lfv[M]['tex']+' '+_tex_lfv[''.join(ll)]
    _process_taxonomy = r'Process :: quarkonium lepton decays :: $V\to \ell^+\ell^-$ :: $' + _process_tex + r"$"
    _obs_name = "BR("+M+''.join(ll)+")"
    _obs = Observable(_obs_name)
    _obs.set_description(r"Branching ratio of $"+_process_tex+r"$")
    _obs.tex = r"$\text{BR}(" + _process_tex+r")$"
    _obs.add_taxonomy(_process_taxonomy)
    return _obs_name

for M in _hadr_lfv:
    for ll in [('e','mu'), ('mu','e'), ('e','tau'), ('tau','e'), ('mu','tau'), ('tau','mu')]:
        _obs_name = _define_obs_V_ll(M, ll)
        Prediction(_obs_name, Vll_br_func(_hadr_lfv[M]['V'], ll[0], ll[1]))
    for ll in [('e','mu'), ('e','tau'), ('mu','tau')]:
        # Combined l1+ l2- + l2+ l1- lepton flavour violating decays
        _obs_name = _define_obs_V_ll(M, ('{0}{1},{1}{0}'.format(*ll),))
        Prediction(_obs_name, VLL_br_comb_func(_hadr_lfv[M]['V'], ll[0], ll[1]))

# Observable and Prediction instance
# _process_tex = r"J/\psi\to \ell_i\ell_j"
# _process_taxonomy = r'Process :: quarkonium lepton decays :: $q\bar q\to \ell^+\ell^-$ :: $' + _process_tex + r"$"

# _obs_name = "BR(J/psi->ll')"
# _obs = flavio.classes.Observable(_obs_name)
# _obs.set_description(r"Branching ratio of $" + _process_tex + r"$")
# _obs.tex = r"$\text{BR}(" + _process_tex + r")$"
# _obs.add_taxonomy(_process_taxonomy)
# flavio.classes.Prediction(_obs_name, BrJpsi)
