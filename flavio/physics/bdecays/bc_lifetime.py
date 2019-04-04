r"""$B_c$ lifetime"""

import flavio


def tau_Bc(wc_obj, par):
    r"""Lifetime of the $B_c$ meson based on the SM OPE estimate plus
    the NP contribution to leptonic decays."""
    tau_SM = par['tau_Bc_SM']
    tau_exp = par['tau_Bc']
    tau_lnu = 0
    wc_sm = flavio.WilsonCoefficients()
    for l in ['e', 'mu', 'tau']:
        _br_SM = flavio.Observable['BR(Bc->{}nu)'.format(l)].prediction_par(par, wc_sm)
        _br_NP = flavio.Observable['BR(Bc->{}nu)'.format(l)].prediction_par(par, wc_obj)
        tau_lnu += (_br_NP - _br_SM) * tau_exp
    return tau_SM + tau_lnu


# Observable and Prediction instance
_process_tex = r"B_c\to X"
_process_taxonomy = r'Process :: $b$ hadron decays :: Lifetimes :: $' + _process_tex + r"$"

_obs_name = "tau_Bc"
_obs = flavio.classes.Observable(_obs_name)
_obs.set_description(r"$B_c$ lifetime")
_obs.tex = r"$\tau_{B_c}$"
_obs.add_taxonomy(_process_taxonomy)
flavio.classes.Prediction(_obs_name, tau_Bc)
