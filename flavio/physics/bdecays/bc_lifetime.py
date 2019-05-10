r"""$B_c$ lifetime"""

import flavio


def tau_Bc(wc_obj, par):
    r"""Lifetime of the $B_c$ meson based on the SM OPE estimate plus
    the NP contribution to leptonic decays."""
    Gamma_SM = 1 / par['tau_Bc_SM']
    Gamma_exp = 1 / par['tau_Bc']
    Gamma_NP = 0
    wc_sm = flavio.WilsonCoefficients()
    for l in ['e', 'mu', 'tau']:
        _br_SM = flavio.Observable['BR(Bc->{}nu)'.format(l)].prediction_par(par, wc_sm)
        _br_NP = flavio.Observable['BR(Bc->{}nu)'.format(l)].prediction_par(par, wc_obj)
        Gamma_NP += (_br_NP - _br_SM) * Gamma_exp
    return 1 / (Gamma_SM + Gamma_NP)


# Observable and Prediction instance
_process_tex = r"B_c\to X"
_process_taxonomy = r'Process :: $b$ hadron decays :: Lifetimes :: $' + _process_tex + r"$"

_obs_name = "tau_Bc"
_obs = flavio.classes.Observable(_obs_name)
_obs.set_description(r"$B_c$ lifetime")
_obs.tex = r"$\tau_{B_c}$"
_obs.add_taxonomy(_process_taxonomy)
flavio.classes.Prediction(_obs_name, tau_Bc)
