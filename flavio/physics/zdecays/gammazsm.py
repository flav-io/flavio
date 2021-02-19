r"""$Z$ partial widths in the SM.

Based on arXiv:1401.2447"""


from math import log
from scipy import constants
import flavio


# units: GeV=hbar=c=1
GeV = constants.giga * constants.eV
s = GeV / constants.hbar
m = s / constants.c
b = 1.e-28 * m**2
pb = constants.pico * b


# Table 5 of 1401.2447
cdict = {
'Gammae,mu': [83.966, -0.047, 0.807, -0.095, -0.01, 0.25, -1.1, 285],
'Gammatau': [83.776, -0.047, 0.806, -0.095, -0.01, 0.25, -1.1, 285],
'Gammanu': [167.157, -0.055, 1.26, -0.19, -0.02, 0.36, -0.1, 503],
'Gammau': [299.936, -0.34, 4.07, 14.27, 1.6, 1.8, -11.1, 1253],
'Gammac': [299.860, -0.34, 4.07, 14.27, 1.6, 1.8, -11.1, 1253],
'Gammad,s': [382.770, -0.34, 3.83, 10.20, -2.4, 0.67, -10.1, 1469],
'Gammab': [375.724, -0.30, -2.28, 10.53, -2.4, 1.2, -10.0, 1458],
'GammaZ': [2494.24, -2.0, 19.7, 58.60, -4.0, 8.0, -55.9, 9267],
'Rl': [20750.9, -8.1, -39, 732.1, -44, 5.5, -358, 11702],
'Rc': [172.23, -0.029, 1.0, 2.3, 1.3, 0.38, -1.2, 37],
'Rb': [215.80, 0.031, -2.98, -1.32, -0.84, 0.035, 0.73, -18],
'sigma0had': [41488.4, 3.0, 60.9, -579.4, 38, 7.3, 85, -86027],
}

# Converting the table to appropriate powers of GeV
units = {
'Gammae,mu': 1e-3, # MeV -> GeV
'Gammatau': 1e-3, # MeV -> GeV
'Gammanu': 1e-3, # MeV -> GeV
'Gammau': 1e-3, # MeV -> GeV
'Gammac': 1e-3, # MeV -> GeV
'Gammad,s': 1e-3, # MeV -> GeV
'Gammab': 1e-3, # MeV -> GeV
'GammaZ': 1e-3, # MeV -> GeV
'Rl': 1e-3,
'Rc': 1e-3,
'Rb': 1e-3,
'sigma0had': pb, # pb
}


def Zobs(name, m_h, m_t, alpha_s, Dalpha, m_Z):
    r"""Expansion formula for $Z$ partial widths according to eq. (28) of
    arXiv:1401.2447.
    """
    flavio.citations.register("Freitas:2014hra")
    L_H = log(m_h / 125.7)
    D_t = (m_t / 173.2)**2 - 1
    D_alpha_s = alpha_s / 0.1184 - 1
    D_alpha = Dalpha / 0.059 - 1
    D_Z = m_Z / 91.1876 - 1
    c = cdict[name]
    return (c[0] + c[1] * L_H + c[2] * D_t + c[3] * D_alpha_s
            + c[4] * D_alpha_s**2 + c[5] * D_alpha_s * D_t
            + c[6] * D_alpha + c[7] * D_Z) * units[name]


def GammaZ_SM(par, f):
    if f in ['e', 'mu']:
        name = 'Gammae,mu'
    elif f in ['d', 's']:
        name = 'Gammad,s'
    elif 'nu' in f:
        name = 'Gammanu'
    else:
        name = 'Gamma' + f
    GSM = Zobs(name, par['m_h'], par['m_t'], par['alpha_s'], 0.059, par['m_Z'])
    return GSM + par['delta_' + name]
