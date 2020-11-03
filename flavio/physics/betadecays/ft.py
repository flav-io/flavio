r"""Functions for nuclear and neutron beta decay effective couplings and $Ft$ values."""


from math import pi, log, sqrt
import flavio
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
    flavio.citations.register("Gonzalez-Alonso:2018omy")
    return 2 * (abs(MF)**2 * (abs(C['V'])**2 + abs(C['S'])**2)
                + abs(MGT)**2 * (abs(C['A'])**2 + abs(C['T'])**2))


def a_xi(C, MF, MGT):
    r"""Correlation coefficients $a\xi$ as function of the effective couplings
    `C`, the Fermi matrix element `MF` and the Gamow-Teller matrix element
    `MGT`."""
    # eq. (16) of arXiv:1803.08732
    # note that C_i' = C_i
    flavio.citations.register("Gonzalez-Alonso:2018omy")
    return 2 * (abs(MF)**2 * (abs(C['V'])**2 - abs(C['S'])**2)
                - 1 / 3 * abs(MGT)**2 * (abs(C['A'])**2 - abs(C['T'])**2))


def a(C, MF, MGT):
    r"""Correlation coefficient $a$ as function of the effective couplings
    `C`, the Fermi matrix element `MF` and the Gamow-Teller matrix element
    `MGT`."""
    return a_xi(C, MF, MGT) / xi(C, MF, MGT)


def b_xi(C, MF, MGT, alpha, Z, s):
    r"""Correlation coefficients $b\xi$ as function of the effective couplings
    `C`, the Fermi matrix element `MF`, the Gamow-Teller matrix element
    `MGT`, the fine structure constant `alpha`, and the nucleon charge `Z`. The sign `s` is + for the electron and - for the positron."""
    # eq. (17) of arXiv:1803.08732
    # note that C_i' = C_i
    flavio.citations.register("Gonzalez-Alonso:2018omy")
    gamma = sqrt(1 - alpha**2 * Z**2)
    return s * 2 * gamma * 2 * (abs(MF)**2 * (C['V'] *  C['S'].conjugate()).real
                                + abs(MGT)**2 * (C['A'] *  C['T'].conjugate()).real)

def dl(Jp, J):
    """Kronecker's delta"""
    if Jp == J:
        return 1
    else:
        return 0

def la(Jp, J):
    """Eq. (A1)"""
    if Jp == J - 1:
        return 1
    elif Jp == J:
        return 1 / (J + 1)
    elif Jp == J + 1:
        return -J / (J + 1)
    else:
        raise ValueError("Invalid input for function `la`")


def A_xi(C, MF, MGT, J, Jf, s):
    r"""Correlation coefficients $A\xi$ as function of the effective couplings
    `C`, the Fermi matrix element `MF`, the Gamow-Teller matrix element
    `MGT`, and the angular momenta of initial and final state nuclei,
    `J` and `Jf`. The sign `s` is + for the electron and - for the
    positron."""
    # note that C_i' = C_i
    return 2 * (s * abs(MGT)**2 * la(Jf, J) * (abs(C['T'])**2 - abs(C['A'])**2)
                + dl(Jf, J) * abs(MF) * abs(MGT) * sqrt(J / (J + 1))
                * (2  * C['S'] * C['T'].conjugate()
                   - 2  * C['V'] * C['A'].conjugate())).real


def B_xi(C, MF, MGT, J, Jf, me_E, s):
    r"""Correlation coefficients $B\xi$ as function of the effective couplings
    `C`, the Fermi matrix element `MF`, the Gamow-Teller matrix element
    `MGT`, and the angular momenta of initial and final state nuclei,
    `J` and `Jf`. `me_E` is the ratio of electron mass and energy.
    The sign `s` is + for the electron and - for the positron."""
    # note that C_i' = C_i
    return 2 * (abs(MGT)**2 * la(Jf, J) * (me_E * 2 * C['T'] * C['A'].conjugate()
                                           + s * (abs(C['T'])**2 + abs(C['A'])**2))
                - dl(Jf, J) * abs(MF) * abs(MGT) * sqrt(J / (J + 1))
                * ((2  * C['S'] * C['T'].conjugate()
                   + 2  * C['V'] * C['A'].conjugate())
                   + s * me_E * (2  * C['S'] * C['A'].conjugate()
                                  + 2  * C['V'] * C['T'].conjugate()))).real


def D_xi(C, MF, MGT, J, Jf):
    r"""Correlation coefficients $D\xi$ as function of the effective couplings
    `C`, the Fermi matrix element `MF`, the Gamow-Teller matrix element
    `MGT`, and the angular momenta of initial and final state nuclei,
    `J` and `Jf`. `me_E` is the ratio of electron mass and energy."""
    # note that C_i' = C_i
    return 2 * (dl(Jf, J) * abs(MF) * abs(MGT) * sqrt(J / (J + 1))
                * (2  * C['S'] * C['T'].conjugate()
                   - 2  * C['V'] * C['A'].conjugate())).imag


def R_xi(C, MF, MGT, J, Jf, s):
    r"""Correlation coefficients $R\xi$ as function of the effective couplings
    `C`, the Fermi matrix element `MF`, the Gamow-Teller matrix element
    `MGT`, and the angular momenta of initial and final state nuclei,
    `J` and `Jf`. The sign `s` is + for the electron and - for the positron."""
    # note that C_i' = C_i
    return 2 * (s * abs(MGT)**2 * la(Jf, J) * 2 * C['T'] * C['A'].conjugate()
                + dl(Jf, J) * abs(MF) * abs(MGT) * sqrt(J / (J + 1))
                * (2  * C['S'] * C['A'].conjugate()
                   - 2  * C['V'] * C['T'].conjugate())).imag


def b(C, MF, MGT, alpha, Z, s):
    r"""Correlation coefficient $b$ as function of the effective couplings
    `C`, the Fermi matrix element `MF`, the Gamow-Teller matrix element
    `MGT`, the fine structure constant `alpha`, and the nucleon charge `Z`."""
    return b_xi(C, MF, MGT, alpha, Z, s) / xi(C, MF, MGT)


def A(C, MF, MGT, J, Jf, s):
    r"""Correlation coefficient $A$ as function of the effective couplings
    `C`, the Fermi matrix element `MF`, the Gamow-Teller matrix element
    `MGT`, and the angular momenta of initial and final state nuclei,
    `J` and `Jf`. The sign `s` is + for the electron and - for the
    positron."""
    return A_xi(C, MF, MGT, J, Jf, s) / xi(C, MF, MGT)


def B(C, MF, MGT, J, Jf, me_E, s):
    r"""Correlation coefficient $B$ as function of the effective couplings
    `C`, the Fermi matrix element `MF`, the Gamow-Teller matrix element
    `MGT`, and the angular momenta of initial and final state nuclei,
    `J` and `Jf`. `me_E` is the ratio of electron mass and energy.
    The sign `s` is + for the electron and - for the positron."""
    return B_xi(C, MF, MGT, J, Jf, me_E, s) / xi(C, MF, MGT)


def D(C, MF, MGT, J, Jf):
    r"""Correlation coefficient $D$ as function of the effective couplings
    `C`, the Fermi matrix element `MF`, the Gamow-Teller matrix element
    `MGT`, and the angular momenta of initial and final state nuclei,
    `J` and `Jf`."""
    return D_xi(C, MF, MGT, J, Jf) / xi(C, MF, MGT)


def R(C, MF, MGT, J, Jf, s):
    r"""Correlation coefficient $R$ as function of the effective couplings
    `C`, the Fermi matrix element `MF`, the Gamow-Teller matrix element
    `MGT`, and the angular momenta of initial and final state nuclei,
    `J` and `Jf`. The sign `s` is + for the electron and - for the positron."""
    return R_xi(C, MF, MGT, J, Jf, s) / xi(C, MF, MGT)


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
    B = b(C, MF, MGT, par['alpha_e'], Z, s=-1)  # s=-1 for beta+ decay
    me_E = nuclei_superallowed[A]['<me/E>']
    Vud = get_ckm(par)[0, 0]
    GF = GFeff(wc_obj, par)
    pre = GF / sqrt(2) * Vud
    ddRp = par['delta_deltaRp_Z2'] * Z**2  # relative uncertainty on \delta R' (universal)
    return (1 + ddRp) * K(par) / Xi * 1 / (1 + B * me_E) / abs(pre)**2


class NeutronObservable:
    def __init__(self, wc_obj, par, me_E):
        self.wc_obj = wc_obj
        self.par = par
        self.me_E = me_E
        self.MF = 1
        self.MGT = sqrt(3)
        self.scale = config['renormalization scale']['betadecay']
        self.C = wc_eff(par, wc_obj, self.scale, nu='e')
        self.s = 1  # electron e- in final state
        self.Z = 0
        self.alpha = par['alpha_e']
        self.J = 1 / 2
        self.Jf = 1 / 2

    def xi(self):
        return xi(self.C, self.MF, self.MGT)

    def a(self):
        return a(self.C, self.MF, self.MGT)

    def b(self):
        return b(self.C, self.MF, self.MGT, self.alpha, self.Z, self.s)

    def A(self):
        return A(self.C, self.MF, self.MGT, self.J, self.Jf, self.s)

    def B(self):
        return B(self.C, self.MF, self.MGT, self.J, self.Jf, self.me_E, self.s)

    def D(self):
        return D(self.C, self.MF, self.MGT, self.J, self.Jf)

    def R(self):
        return R(self.C, self.MF, self.MGT, self.J, self.Jf, self.s)


class Neutron_tau(NeutronObservable):
    def __init__(self, wc_obj, par, me_E):
        super().__init__(wc_obj, par, me_E)

    def __call__(self):
        Vud = get_ckm(self.par)[0, 0]
        GF = GFeff(self.wc_obj, self.par)
        pre = GF / sqrt(2) * Vud
        ft = K(self.par) / self.xi() * 1 / (1 + self.b() * self.me_E) / abs(pre)**2
        fn = self.par['f_n']
        Rp = self.par['deltaRp_n']
        return ft / log(2) / fn / (1 + Rp)


class Neutron_corr(NeutronObservable):
    def __init__(self, wc_obj, par, me_E, coeff):
        super().__init__(wc_obj, par, me_E)
        self.coeff = coeff

    def __call__(self):
        if self.coeff == 'a':
            return self.a()
        elif self.coeff == 'atilde':
            return self.a() / (1 + self.b() * self.me_E)
        if self.coeff == 'b':
            return self.b()
        elif self.coeff == 'A':
            return self.A()
        elif self.coeff == 'Atilde':
            return self.A() / (1 + self.b() * self.me_E)
        elif self.coeff == 'B':
            return self.B()
        elif self.coeff == 'Btilde':
            return self.B() / (1 + self.b() * self.me_E)
        elif self.coeff == 'lambdaAB':
            _A = self.A()
            _B = self.B()
            return (_A - _B) / (_A + _B)
        elif self.coeff == 'D':
            return self.D()
        elif self.coeff == 'R':
            return self.R()


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
for _A, _Ad in nuclei_superallowed.items():
    Dd = get_daughter(_A)
    _process_tex = _Ad['tex'] + r"\to " + Dd['tex'] + r"\,e^+\nu_e"
    _process_taxonomy = r'Process :: Nucleon decays :: Beta decays :: Superallowed $0^+\to 0^+$ decays :: $' + _process_tex + r"$"
    _obs_name = "Ft(" + _A + ")"
    _obs = Observable(_obs_name)
    _obs.set_description(r"$\mathcal Ft$ value of $" + _Ad['tex'] + r"$ beta decay")
    _obs.tex = r"$\mathcal{F}t(" + _Ad['tex'] + r")$"
    _obs.add_taxonomy(_process_taxonomy)
    Prediction(_obs_name, Ft_fct(_A))


_process_tex = r"n\to p^+ e^-\bar\nu_e"
_process_taxonomy = r'Process :: Nucleon decays :: Beta decays :: Neutron decay :: $' + _process_tex + r"$"
_obs_name = "tau_n"
_obs = Observable(_obs_name, arguments=['me_E'])
_obs.set_description(r"Neutron lifetime")
_obs.tex = r"$\tau_n$"
_obs.add_taxonomy(_process_taxonomy)
func = lambda wc_obj, par, me_E: Neutron_tau(wc_obj, par, me_E)()
Prediction(_obs_name, func)


# coefficients that don't depend on me/E
coeffs = {
    'a': 'a_n',
    'A': 'A_n',
    'D': 'D_n',
    'R': 'R_n',
}

# coefficients that depend on me/E
coeffs_mE = {
    'atilde': r'\tilde{a}_n',
    'b': 'b_n',
    'Atilde': r'\tilde{A}_n',
    'B': 'B_n', 'Btilde': r'\tilde{B}_n',
    'lambdaAB': r'\lambda_{AB}',
}


def make_obs_neutron_corr(coeff, me_E=False):
    _process_tex = r"n\to p^+ e^-\bar\nu_e"
    _process_taxonomy = r'Process :: Nucleon decays :: Beta decays :: Neutron decay :: $' + _process_tex + r"$"
    _obs_name = coeff + "_n"
    if me_E:
            _obs = Observable(_obs_name, arguments=['me_E'])
    else:
        _obs = Observable(_obs_name)
    _obs.set_description(r"Correlation coefficient $" + tex + r"$ in neutron beta decay")
    _obs.tex = r"$" + tex + r"$"
    _obs.add_taxonomy(_process_taxonomy)
    if me_E:
        func = lambda wc_obj, par, me_E: Neutron_corr(wc_obj, par, me_E, coeff)()
    else:
        func = lambda wc_obj, par: Neutron_corr(wc_obj, par, None, coeff)()
    Prediction(_obs_name, func)


for coeff, tex in coeffs.items():
    make_obs_neutron_corr(coeff, me_E=False)


for coeff, tex in coeffs_mE.items():
    make_obs_neutron_corr(coeff, me_E=True)
