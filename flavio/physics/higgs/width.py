r"""Functions for the total Higgs width."""

from . import decay
from math import pi


# SM Higgs BRs, taken from Higgs XSWG for m_h=125.10 GeV
BR_SM = {
    'bb': 0.5807,
    'WW': 0.2154,
    'ZZ': 0.02643,
    'gg': 0.08179,
    'tautau': 0.06256,
}


# SM total width over Higgs mass
Gamma_rel_SM = 4.101e-3 / 125.10


def Gamma_rel_ff(y_ff, Nc):
    """Higgs partial width to massless fermion pair divided by Higgs mass
    as function of the effective Yukawa coupling and "colour" multiplicity."""
    return abs(y_ff)**2 * Nc / (8 * pi)


def Gamma_h(par, C):
    """Higgs total width, normalized to its SM value.

    For the 5 most frequent SM decay modes, only the interference terms
    of SM and NP are taken into account.

    Additionally, squared contributions are included for contributions from
    modified couplings to the four lightest quarks, that become relevant
    for very nonstandard light quark Yukawas.
    """
    R_bb =  (decay.h_bb(C) - 1) * BR_SM['bb']
    R_tautau =  (decay.h_tautau(C) - 1) * BR_SM['tautau']
    R_WW =  (decay.h_ww(C) - 1) * BR_SM['WW']
    R_ZZ =  (decay.h_zz(C) - 1) * BR_SM['ZZ']
    R_gg =  (decay.h_gg(C) - 1) * BR_SM['gg']
    R_Gamma_SM = 1
    R_Gamma_linear = R_bb + R_tautau + R_WW + R_ZZ + R_gg
    R_Gamma_quadratic  = 0
    for q in ['u', 'd']:
        for i in [1, 2]:
            # here the shift in G_F is neglected
            y_eff = 1 / par['GF'] / 2 * C['{}phi_{}{}'.format(q, i, i)]
            R_Gamma_quadratic += Gamma_rel_ff(y_eff, Nc=3) / Gamma_rel_SM
    return R_Gamma_SM + R_Gamma_linear + R_Gamma_quadratic
