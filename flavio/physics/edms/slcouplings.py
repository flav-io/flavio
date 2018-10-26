"""Functions for effective semileptonic couplings."""

from flavio.physics.edms.common import proton_charges


def CS(wc, par, scale, Z, N):
    """Scalar electron-nucleon coupling."""
    me = proton_charges(par, scale)
    CSu = -(wc['CSRR_eeuu'].imag + wc['CSRL_eeuu'].imag)
    CSd = -(wc['CSRR_eedd'].imag + wc['CSRL_eedd'].imag)
    CS0 = me['gS_u+d'] * (CSu + CSd)
    CS1 = me['gS_u-d'] * (CSu - CSd)
    return CS0 + (Z - N) / (Z + N) * CS1
