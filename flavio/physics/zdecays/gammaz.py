r"""$Z$ pole observables beyond the SM."""

import flavio
from flavio.physics.zdecays import gammazsm, smeftew
from math import sqrt, pi



def GammaZ_NP(par, Nc, gV_SM, d_gV, gA_SM, d_gA):
    GF, mZ = par['GF'], par['m_Z']
    return (
        sqrt(2) * GF * mZ**3 / (3 * pi) * Nc * (
            2*(gV_SM*d_gV).real + 2*(gA_SM*d_gA).real
            + abs(d_gV)**2 + abs(d_gA)**2
        )
    )


def GammaZ(wc_obj, par, f1, f2):
    scale = flavio.config['renormalization scale']['zdecays']
    wc_dict = wc_obj.get_wcxf(sector='all', scale=scale, par=par,
                              eft='SMEFT', basis='Warsaw')
    Nc = smeftew._QN[f1]['Nc']
    if f1 == f2:
        gV_SM = smeftew.gV_SM(f1, par)
        gA_SM = smeftew.gA_SM(f1, par)
        GSM = gammazsm.GammaZ_SM(par, f1)
    else:
        gV_SM = 0
        gA_SM = 0
        GSM = 0
    d_gV = smeftew.d_gV(f1, f2, par, wc_dict)
    d_gA = smeftew.d_gA(f1, f2, par, wc_dict)
    GNP = GammaZ_NP(par, Nc, gV_SM, d_gV, gA_SM, d_gA)
    return GSM + GNP


def GammaZ_fct(f1, f2):
    def fu(wc_obj, par):
        return GammaZ(wc_obj, par, f1, f2)
    return fu


def BRZ_fct(f1, f2):
    def fu(wc_obj, par):
        return par['tau_Z'] * GammaZ(wc_obj, par, f1, f2)
    return fu

def BRZ_fct_av(f1, f2):
    def fu(wc_obj, par):
        return par['tau_Z'] * (GammaZ(wc_obj, par, f1, f2) + GammaZ(wc_obj, par, f2, f1))
    return fu


def GammaZnu(wc_obj, par):
    # sum over all 9 possibilities (they are exp. indistinguishable)
    lep = ['e', 'mu', 'tau']
    return sum([GammaZ(wc_obj, par, 'nu' + l1, 'nu' + l2) for l1 in lep for l2 in lep]) / 3.


def Gammal(wc_obj, par):
    # sum only over the three possibilities with equal leptons (they are exp. distinguished)
    return sum([GammaZ(wc_obj, par, l, l) for l in ['e', 'mu', 'tau']]) / 3.


def Gammahad(wc_obj, par):
    # sum over all 9 possibilities (they are exp. not distinguished)
    Gu =  sum([GammaZ(wc_obj, par, q1, q2) for q1 in 'uc' for q2 in 'uc'])
    Gd =  sum([GammaZ(wc_obj, par, q1, q2) for q1 in 'dsb' for q2 in 'dsb'])
    return Gu + Gd


def sigmahad(wc_obj, par):
    # e+e- hadronic Z production xsec
    return 12 * pi / par['m_Z']**2 / Gammatot(wc_obj, par)**2 * GammaZ(wc_obj, par, 'e', 'e') * Gammahad(wc_obj, par)


def Gammatot(wc_obj, par):
    # total Z width
    return Gammahad(wc_obj, par) + Gammal(wc_obj, par) * 3 + GammaZnu(wc_obj, par) * 3


def Rl(wc_obj, par):
    return Gammahad(wc_obj, par) / Gammal(wc_obj, par)


def Rq(f):
    def _Rq(wc_obj, par):
        return GammaZ(wc_obj, par, f, f) / Gammahad(wc_obj, par)
    return _Rq


def Remutau(f):
    def _Remutau(wc_obj, par):
        return Gammahad(wc_obj, par) / GammaZ(wc_obj, par, f, f)
    return _Remutau


_leptons = {'e': ' e', 'mu': r'\mu', 'tau': r'\tau'}
_uquarks = {'u': ' u', 'c': ' c', 'c': ' c'}
_dquarks = {'d': ' d', 's': ' s', 'b': ' b'}


for _f in (_leptons, _uquarks, _dquarks):
    for f, tex in _f.items():
        if f in _leptons:
            _process_tex = r"Z^0\to {}^+{}^-".format(tex, tex)
        else:
            _process_tex = r"Z^0\to{}\bar{}".format(tex, tex)

        _obs_name = "Gamma(Z->{})".format(2 * f)
        _obs = flavio.classes.Observable(_obs_name)
        _obs.tex = r"$\Gamma(" + _process_tex + r")$"
        _obs.set_description(r"Partial width of $" + _process_tex + r"$")
        _obs.add_taxonomy(r'Process :: $Z^0$ decays :: Flavour conserving decays :: $' + _process_tex + r"$")
        flavio.classes.Prediction(_obs_name, GammaZ_fct(f, f))

        _obs_name = "R_{}".format(f)
        _obs = flavio.classes.Observable(_obs_name)
        _obs.tex = r"$R_{}^0$".format(tex)
        _obs.add_taxonomy(r'Process :: $Z^0$ decays :: Flavour conserving decays :: $' + _process_tex + r"$")
        if f in _leptons:
            _obs.set_description(r"Ratio of $Z^0$ partial widths to hadrons vs. ${}$ pairs".format(tex))
            flavio.classes.Prediction(_obs_name, Remutau(f))
        else:
            _obs.set_description(r"Ratio of $Z^0$ partial widths to ${}$ pairs vs. all hadrons".format(tex))
            flavio.classes.Prediction(_obs_name, Rq(f))


# LFV Z decays
for (f1, f2) in [('e', 'mu'), ('e', 'tau'), ('mu', 'tau'), ]:
    tex1 = _leptons[f1]
    tex2 = _leptons[f2]
    _obs_name = "BR(Z->{}{})".format(f1, f2)
    _obs = flavio.classes.Observable(_obs_name)
    _process_tex = r"Z^0\to {}^\pm{}^\mp".format(tex1, tex2)
    _obs.tex = r"$\text{BR}(" + _process_tex + r")$"
    _obs.set_description(r"Branching ratio of $" + _process_tex + r"$")
    _obs.add_taxonomy(r'Process :: $Z^0$ decays :: FCNC decays :: $' + _process_tex + r"$")
    flavio.classes.Prediction(_obs_name, BRZ_fct_av(f1, f2))


_obs_name = "GammaZ"
_obs = flavio.classes.Observable(_obs_name)
_obs.tex = r"$\Gamma_Z$"
_obs.set_description(r"Total width of the $Z^0$ boson")
for _f in (_leptons, _uquarks, _dquarks):
    for f, tex in _f.items():
        if f in _leptons:
            _process_tex = r"Z^0\to {}^+{}^-".format(tex, tex)
        else:
            _process_tex = r"Z^0\to{}\bar{}".format(tex, tex)
        _obs.add_taxonomy(r'Process :: $Z^0$ decays :: Flavour conserving decays :: $' + _process_tex + r"$")
flavio.classes.Prediction(_obs_name, Gammatot)

_obs_name = "sigma_had"
_obs = flavio.classes.Observable(_obs_name)
_obs.tex = r"$\sigma_\text{had}^0$"
_obs.set_description(r"$e^+e^-\to Z^0$ hadronic pole cross-section")
for _f in (_uquarks, _dquarks):
    for f, tex in _f.items():
        _process_tex = r"Z^0\to{}\bar{}".format(tex, tex)
        _obs.add_taxonomy(r'Process :: $Z^0$ decays :: Flavour conserving decays :: $' + _process_tex + r"$")
flavio.classes.Prediction(_obs_name, sigmahad)


_obs_name = "Gamma(Z->nunu)"
_obs = flavio.classes.Observable(_obs_name)
_process_tex = r"Z^0\to\nu\bar\nu"
_obs.tex = r"$\Gamma(" + _process_tex + r")$"
_obs.set_description(r"Partial width of $" + _process_tex + r"$, averaged over neutrino flavours")
_obs.add_taxonomy(r'Process :: $Z^0$ decays :: Flavour conserving decays :: $' + _process_tex + r"$")
flavio.classes.Prediction(_obs_name, GammaZnu)


_obs_name = "R_l"
_obs = flavio.classes.Observable(_obs_name)
_obs.tex = r"$R_l^0$"
_obs.set_description(r"Ratio of $Z^0$ partial widths to hadrons vs. leptons, averaged over lepton flavours")
for l in [' e', r'\mu', r'\tau']:
    _obs.add_taxonomy(r"Process :: $Z^0$ decays :: Flavour conserving decays :: $Z^0\to {}^+{}^-$".format(l, l))
flavio.classes.Prediction(_obs_name, Rl)
