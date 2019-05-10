r"""Functions for $\Lambda\to p\ell\nu decay"""

import flavio
from flavio.physics.bdecays.wilsoncoefficients import get_wceff_fccc_std


def g1_f1(wc_obj, par, lep, nu):
    scale = flavio.config['renormalization scale']['kdecays']
    ms = flavio.physics.running.running.get_ms(par, scale)
    wc = get_wceff_fccc_std(wc_obj, par, 'su', lep, nu, ms, scale, nf=3)
    f1 = par['Lambda->p f_1(0)'] * (wc['VL'] + wc['VR']).real
    g1 = par['Lambda->p g_1(0)'] * (wc['VL'] - wc['VR']).real
    return g1 / f1


def g1_f1_fct(lep):
    def f(wc_obj, par):
        return g1_f1(wc_obj, par, lep=lep, nu=lep)
    return f


_tex = {'e': 'e', 'mu': r'\mu', 'l': r'\ell'}
for l in ['e', 'mu', 'l']:
    _process_tex = r'\Lambda\to p'+_tex[l]+r"^+\nu"
    _process_taxonomy = r'Process :: $s$ hadron decays :: Semi-leptonic tree-level decays :: $\Lambda\to p\ell\nu$ :: $' + _process_tex + r"$"

    _obs_name = "g1/f1(Lambda->p"+l+"nu)"
    _obs = flavio.classes.Observable(_obs_name)
    _obs.set_description(r"Ratio of axial to vector form factor in $" + _process_tex + r"$")
    _obs.tex = r"$\frac{g_1(0)}{f_1(0)}(" + _process_tex + r")$"
    _obs.add_taxonomy(_process_taxonomy)
    flavio.classes.Prediction(_obs_name, g1_f1_fct(l))
