r"""$W\to\ell\nu$ decays."""


import flavio
from flavio.physics.zdecays import smeftew
from math import sqrt, pi


def gWl_SM(f1, f2, par):
    if f1 == f2:
        return 1 / 2   # using neutrino flavour eigenstates
    else:
        return 0

def gWq_SM(f1, f2, par):
    i = smeftew._sectors['u'][f1]
    j = smeftew._sectors['d'][f2]
    V = flavio.physics.ckm.get_ckm(par)
    return V[i - 1, j - 1] / 2

def gWf_SM(f1, f2, par):
    if f1 in smeftew._sectors['l']:
        return gWl_SM(f1, f2, par)
    else:
        return gWq_SM(f1, f2, par)

def GammaWq_SM(f1, f2, par):
    return 4 * abs(gWq_SM(f1, f2, par))**2 * par['GammaW_had'] / 2

def GammaWl_SM(f1, f2, par):
    return 4 * abs(gWl_SM(f1, f2, par))**2 * par['GammaW_lep'] / 3

def GammaW_SM(f1, f2, par):
    if f1 in smeftew._sectors['l']:
        return GammaWl_SM(f1, f2, par)
    if f1 in smeftew._sectors['u']:
        return GammaWq_SM(f1, f2, par)


def GammaW_NP(par, Nc, gW_SM, d_gW):
    GF, mW = par['GF'], par['m_W']
    return (sqrt(2) * GF * mW**3 / (3 * pi) * Nc
            * (2*(gW_SM*d_gW).real + abs(d_gW)**2))


def GammaW(wc_obj, par, f1, f2):
    scale = flavio.config['renormalization scale']['wdecays']
    wc_dict = wc_obj.get_wcxf(sector='all', scale=scale, par=par,
                              eft='SMEFT', basis='Warsaw')
    Nc = smeftew._QN[f1]['Nc']
    gW_SM = gWf_SM(f1, f2, par)
    GSM = GammaW_SM(f1, f2, par)
    d_gW = smeftew.d_gW(f1, f2, par, wc_dict)
    GNP = GammaW_NP(par, Nc, gW_SM, d_gW)
    return GSM + GNP


def Gammalep(wc_obj, par):
    lep = ['e', 'mu', 'tau']
    return sum([GammaW(wc_obj, par, l1, l2) for l1 in lep for l2 in lep])

def Gammahad(wc_obj, par):
    return sum([GammaW(wc_obj, par, q1, q2) for q1 in 'uc' for q2 in 'dsb'])

def Gammatot(wc_obj, par):
    return Gammalep(wc_obj, par) + Gammahad(wc_obj, par)


def BRWlnu_fct(l):
    def fu(wc_obj, par):
        lep = ['e', 'mu', 'tau']
        # use calculated lifetime, which is more precise than measured one!
        return sum([GammaW(wc_obj, par, l, nu) for nu in lep]) / Gammatot(wc_obj, par)
    return fu


def RWlnu_fct(l1, l2):
    def fu(wc_obj, par):
        lep = ['e', 'mu', 'tau']
        # use calculated lifetime, which is more precise than measured one!
        num = sum([GammaW(wc_obj, par, l1, nu) for nu in lep])
        den = sum([GammaW(wc_obj, par, l2, nu) for nu in lep])
        return num / den
    return fu


def RWcX(wc_obj, par):
    q = ['d', 's', 'b']
    return sum([GammaW(wc_obj, par, 'c', d) for d in q]) / Gammahad(wc_obj, par)


_leptons = {'e': ' e', 'mu': r'\mu', 'tau': r'\tau'}
_uquarks = {'u': ' u', 'c': ' c', 'c': ' c'}
_dquarks = {'d': ' d', 's': ' s', 'b': ' b'}


for f, tex in _leptons.items():
    _process_tex = r"W^\pm\to {}^\pm\nu".format(tex)
    _obs_name = "BR(W->{}nu)".format(f)
    _obs = flavio.classes.Observable(_obs_name)
    _obs.tex = r"$\text{BR}(" + _process_tex + r")$"
    _obs.set_description(r"Branching ratio of $" + _process_tex + r"$, summed over neutrino flavours")
    _obs.add_taxonomy(r'Process :: $W^\pm$ decays :: Leptonic decays :: $' + _process_tex + r"$")
    flavio.classes.Prediction(_obs_name, BRWlnu_fct(f))

for (f1, f2) in [('tau', 'e'), ('mu', 'e'), ('tau', 'mu')]:
    tex1 = _leptons[f1]
    tex2 = _leptons[f2]
    _process_tex = r"W^\pm\to \ell^\pm\nu".format(tex2)
    _process_tex1 = r"W^\pm\to {}^\pm\nu".format(tex1)
    _process_tex2 = r"W^\pm\to {}^\pm\nu".format(tex2)
    _obs_name = "R{}{}(W->lnu)".format(f1, f2)
    _obs = flavio.classes.Observable(_obs_name)
    _obs.tex = r"$\text{R}_{" + tex1 + " " + tex2 + "}(" + _process_tex + r")$"
    _obs.set_description(r"Ratio of branching ratio of $" + _process_tex1 + r"$ and  $" + _process_tex2 + r"$, individually summed over neutrino flavours")
    _obs.add_taxonomy(r'Process :: $W^\pm$ decays :: Leptonic decays :: $' + _process_tex1 + r"$")
    _obs.add_taxonomy(r'Process :: $W^\pm$ decays :: Leptonic decays :: $' + _process_tex2 + r"$")
    flavio.classes.Prediction(_obs_name, RWlnu_fct(f1, f2))


_process_tex = r"W^+\to cX"
_obs_name = "R(W->cX)"
_obs = flavio.classes.Observable(_obs_name)
_obs.tex = r"$\text{R}(" + _process_tex + r")$"
_obs.set_description(r"Ratio of partial width of $" + _process_tex + r"$, $X=\bar d, \bar s, \bar b$ over the hadronic $W$ width")
_obs.add_taxonomy(r'Process :: $W^\pm$ decays :: Hadronic decays :: $' + _process_tex + r"$")
flavio.classes.Prediction(_obs_name, RWcX)


_obs_name = "GammaW"
_obs = flavio.classes.Observable(_obs_name)
_obs.tex = r"$\Gamma_W$"
_obs.set_description(r"Total width of the $W^\pm$ boson")
for f, tex in _leptons.items():
    _process_tex = r"W^\pm\to {}^\pm\nu".format(tex)
    _obs.add_taxonomy(r'Process :: $W^\pm$ decays :: Leptonic decays :: $' + _process_tex + r"$")
flavio.classes.Prediction(_obs_name, Gammatot)
