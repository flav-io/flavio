r"""$Z^0$ decay asymmetries."""


import flavio
from flavio.physics.zdecays.smeftew import d_gV, d_gA, gV_SM, gA_SM


def Af(gV, gA):
    r = (gV / gA).real  # chop imaginary part (from numerical noise, should be real)
    return 2 * r / (1 + r**2)


def AFBf(gVe, gAe, gVf, gAf):
    A = Af(gVf, gAf)
    Ae = Af(gVe, gAe)
    return 3 / 4 * Ae * A


def get_gV_gA(wc_obj, par, f):
        scale = flavio.config['renormalization scale']['zdecays']
        C = wc_obj.get_wcxf(sector='all', scale=scale, par=par,
                            eft='SMEFT', basis='Warsaw')
        gV = gV_SM(f, par) + d_gV(f, f, par, C)
        gA = gA_SM(f, par) + d_gA(f, f, par, C)
        return gV, gA


def Af_fct(f):
    def _f(wc_obj, par):
        gV, gA = get_gV_gA(wc_obj, par, f)
        return Af(gV, gA)
    return _f


def AFBf_fct(f):
    def _f(wc_obj, par):
        gV, gA = get_gV_gA(wc_obj, par, f)
        gVe, gAe = get_gV_gA(wc_obj, par, 'e')
        return AFBf(gVe, gAe, gV, gA)
    return _f


_leptons = {'e': ' e', 'mu': r'\mu', 'tau': r'\tau'}
_uquarks = {'u': ' u', 'c': ' c', 'c': ' c'}
_dquarks = {'d': ' d', 's': ' s', 'b': ' b'}


for _f in (_leptons, _uquarks, _dquarks):
    for f, tex in _f.items():
        if f in _leptons:
            _process_tex = r"Z^0\to {}^+{}^-".format(tex, tex)
        else:
            _process_tex = r"Z^0\to{}\bar{}".format(tex, tex)
        _process_taxonomy = r'Process :: $Z^0$ decays :: Flavour conserving decays :: $' + _process_tex + r"$"

        # A_f
        _obs_name = "A(Z->{})".format(2 * f)
        _obs = flavio.classes.Observable(_obs_name)
        _obs.tex = r"$A_{}$".format(tex)
        _obs.set_description(r"Asymmetry parameter in $" + _process_tex + r"$")
        _obs.add_taxonomy(_process_taxonomy)
        flavio.classes.Prediction(_obs_name, Af_fct(f))

        # AFB_f
        _obs_name = "AFB(Z->{})".format(2 * f)
        _obs = flavio.classes.Observable(_obs_name)
        _obs.tex = r"$A_\text{{FB}}^{{0,{}}}$".format(tex)
        _obs.set_description(r"Forward-backward asymmetry in $" + _process_tex + r"$")
        _obs.add_taxonomy(_process_taxonomy)
        flavio.classes.Prediction(_obs_name, AFBf_fct(f))
