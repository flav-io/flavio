from math import pi, log, sqrt
from flavio.config import config
from flavio.physics.betadecays.common import wc_eff
from flavio.physics.ckm import get_ckm
from flavio.physics.taudecays.taulnunu import GFeff
from flavio.physics import elements
from flavio.classes import Observable, Prediction
import re


def xi(C, MF, MGT):
    r"""Correlation coefficient $\xi$ as function of the effective couplings
    `C`, the Fermi matrix element `MF` and the Gamow-Teller matrix element
    `MGT`."""
    # eq. (15) of arXiv:1803.08732
    # note that C_i' = C_i
    return (2 * abs(MF)**2 * (abs(C['V'])**2 + abs(C['S'])**2)
            + 2 * abs(MGT)**2 * (abs(C['A'])**2 + abs(C['T'])**2))


def a_xi(C, MF, MGT):
    r"""Correlation coefficients $a\xi$ as function of the effective couplings
    `C`, the Fermi matrix element `MF` and the Gamow-Teller matrix element
    `MGT`."""
    # eq. (16) of arXiv:1803.08732
    # note that C_i' = C_i
    return (2 * abs(MF)**2 * (abs(C['V'])**2 - abs(C['S'])**2)
            + 2 / 3 * abs(MGT)**2 * (abs(C['A'])**2 - abs(C['T'])**2))


def a(C, MF, MGT, alpha, Z):
    r"""Correlation coefficient $a$ as function of the effective couplings
    `C`, the Fermi matrix element `MF` and the Gamow-Teller matrix element
    `MGT`."""
    return a_xi(C, MF, MGT, alpha, Z) / xi(C, MF, MGT)


def b_xi(C, MF, MGT, alpha, Z):
    r"""Correlation coefficients $b\xi$ as function of the effective couplings
    `C`, the Fermi matrix element `MF`, the Gamow-Teller matrix element
    `MGT`, the fine structure constant `alpha`, and the nucleon charge `Z`."""
    # eq. (17) of arXiv:1803.08732
    # note that C_i' = C_i
    gamma = sqrt(1 - alpha**2 * Z**2)
    return 2 * gamma *(2 * abs(MF)**2 * (C['V'] *  C['S'].conjugate()).real
                       + 2 * abs(MGT)**2 * (C['A'] *  C['T'].conjugate()).real)


def b(C, MF, MGT, alpha, Z):
    r"""Correlation coefficient $b$ as function of the effective couplings
    `C`, the Fermi matrix element `MF`, the Gamow-Teller matrix element
    `MGT`, the fine structure constant `alpha`, and the nucleon charge `Z`."""
    return b_xi(C, MF, MGT, alpha, Z) / xi(C, MF, MGT)


def K(par):
    me = par['m_e']
    return 2 * pi**3 * log(2) / me**5


# <me/E> from Table 4 of arXiv:1803.08732
nuclei_superallowed = {
    # superallowed 0+->0+
    '10C':  {'Z': 6,  '<me/E>': 0.619, 'tex': r'{}^{10}\text{C}'},
    '14O':  {'Z': 8,  '<me/E>': 0.438, 'tex': r'{}^{14}\text{O}'},
    '22Mg': {'Z': 12, '<me/E>': 0.310, 'tex': r'{}^{22}\text{Mg}'},
    '26mAl':{'Z': 13, '<me/E>': 0.300, 'tex': r'{}^{26m}\text{Al}'},
    '34Cl': {'Z': 17, '<me/E>': 0.234, 'tex': r'{}^{34}\text{Cl}'},
    '34Ar': {'Z': 18, '<me/E>': 0.212, 'tex': r'{}^{34}\text{Ar}'},
    '38mK': {'Z': 19, '<me/E>': 0.213, 'tex': r'{}^{38m}\text{K}'},
    '38Ca': {'Z': 20, '<me/E>': 0.195, 'tex': r'{}^{38}\text{Ca}'},
    '42Sc': {'Z': 21, '<me/E>': 0.201, 'tex': r'{}^{42}\text{Sc}'},
    '46V':  {'Z': 23, '<me/E>': 0.183, 'tex': r'{}^{46}\text{V}'},
    '50Mn': {'Z': 25, '<me/E>': 0.169, 'tex': r'{}^{50}\text{Mn}'},
    '54Co': {'Z': 27, '<me/E>': 0.157, 'tex': r'{}^{54}\text{Co}'},
    '62Ga': {'Z': 31, '<me/E>': 0.141, 'tex': r'{}^{62}\text{Ga}'},
    '74Rb': {'Z': 37, '<me/E>': 0.125, 'tex': r'{}^{74}\text{Rb}'},
}


def Ft_superallowed(par, wc_obj, A):
    r"""Corrected $\mathcal{F}t$ value of the beta decay of isotope `A`."""
    MF = sqrt(2)
    MGT = 0
    Z = nuclei_superallowed[A]['Z']
    scale = config['renormalization scale']['betadecay']
    C = wc_eff(par, wc_obj, scale, nu='e')
    Xi = xi(C, MF, MGT)
    B = b(C, MF, MGT, par['alpha_e'], Z)
    me_E = nuclei_superallowed[A]['<me/E>']
    Vud = get_ckm(par)[0, 0]
    GF = GFeff(wc_obj, par)
    pre = GF / sqrt(2) * Vud
    ddRp = par['delta_deltaRp_Z2'] * Z**2  # relative uncertainty on \delta R' (universal)
    return (1 + ddRp) * K(par) / Xi * 1 / (1 + B * me_E) / abs(pre)**2


def tau_n(wc_obj, par, me_E):
    MF = 1
    MGT = sqrt(3)
    scale = config['renormalization scale']['betadecay']
    C = wc_eff(par, wc_obj, scale, nu='e')
    Xi = xi(C, MF, MGT)
    B = b(C, MF, MGT, par['alpha_e'], Z=0)
    from flavio.physics.units import s
    Vud = get_ckm(par)[0, 0]
    GF = GFeff(wc_obj, par)
    pre = GF / sqrt(2) * Vud
    ft = K(par) / Xi * 1 / (1 + B * me_E) / abs(pre)**2
    fn = par['f_n']
    Rp = par['deltaRp_n']
    return ft / log(2) / fn / (1 + Rp)


# Closures for prediction instances
def Ft_fct(A):
    def _(wc_obj, par):
        return Ft_superallowed(par, wc_obj, A)
    return _


def get_daughter(nuclide):
    r"""Get the symbol and tex code of the daughter nuclide."""
    A = re.search(r'\d+', nuclide).group()
    symbol = re.search(r'[A-Z].*', nuclide).group()
    Z = elements.Z(symbol)
    daughter_symbol = elements.symbol(Z - 1)
    return {'name': '{}{}'.format(A, daughter_symbol),
            'tex': r'{{}}^{{{}}}\text{{{}}}'.format(A, daughter_symbol)}


# Observable and Prediction instances
for A, Ad in nuclei_superallowed.items():
    Dd = get_daughter(A)
    _process_tex = Ad['tex'] + r"\to " + Dd['tex'] + r"\,e^+\nu_e"
    _process_taxonomy = r'Process :: Nucleon decays :: Beta decays :: Superallowed $0^+\to 0^+$ decays :: $' + _process_tex + r"$"
    _obs_name = "Ft(" + A + ")"
    _obs = Observable(_obs_name)
    _obs.set_description(r"$\mathcal Ft$ value of $" + Ad['tex'] + r"$ beta decay")
    _obs.tex = r"$\mathcal{F}t(" + Ad['tex'] + r")$"
    _obs.add_taxonomy(_process_taxonomy)
    Prediction(_obs_name, Ft_fct(A))


_process_tex = r"n\to p^+ e^-\bar\nu_e"
_process_taxonomy = r'Process :: Nucleon decays :: Beta decays :: Neutron decay :: $' + _process_tex + r"$"
_obs_name = "tau_n"
_obs = Observable(_obs_name, arguments=['me_E'])
_obs.set_description(r"Neutron lifetime")
_obs.tex = r"$\tau_n$"
_obs.add_taxonomy(_process_taxonomy)
Prediction(_obs_name, tau_n)
