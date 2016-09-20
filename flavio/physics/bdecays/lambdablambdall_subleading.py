import flavio
import numpy as np
from flavio.classes import AuxiliaryQuantity, Implementation
from flavio.physics.bdecays.common import meson_quark
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict
from flavio.physics.common import conjugate_par, conjugate_wc, add_dict
from flavio.config import config

# auxiliary function to construct transversity_amps_deltaC7, transversity_amps_deltaC9
def _transversity_amps_deltaC(q2, deltaC, C_name, par):
    scale = flavio.config['renormalization scale']['lambdab']
    mLb = par['m_Lambdab']
    mL = par['m_Lambda']
    mb = flavio.physics.running.running.get_mb(par, scale)
    ff = flavio.physics.bdecays.lambdablambdall.get_ff(q2, par)
    N = flavio.physics.bdecays.lambdablambdall.prefactor(q2, par, scale)
    ha = flavio.physics.bdecays.lambdablambdall.helicity_amps(q2, mLb, mL, ff)
    wc  = {'7': 0, '7p': 0, 'v': 0, 'a': 0, 's': 0, 'p': 0, 't': 0,'vp': 0, 'ap': 0, 'sp': 0, 'pp': 0, 'tp': 0, }
    wc[C_name] = deltaC
    return flavio.physics.bdecays.lambdablambdall.transverity_amps(ha, q2, mLb, mL, mb, 0, wc, N)


def transversity_amps_deltaC7(q2, deltaC7_dict, par):
    r"""A function returning a contribution to the transversity amplitudes in
    $\Lambda_b\to\Lambda\ell^+\ell^-$ coming from an effective transversity-dependent shift of
    the Wilson coefficient $C_7(\mu_b)$. This can be used to parametrize
    residual uncertainties due to subleading non-factorizable hadronic effects.

    The input dictionary `deltaC7_dict` should be of the form

    `{ 'perp0': deltaC7_perp0, 'para0': deltaC7_para0, 'perp1': deltaC7_perp1, 'para1': deltaC7_para1}`
    """
    ta = {}
    for amp in ['perp0', 'para0', 'perp1', 'para1']:
        for X in ['L', 'R']:
            ta[(amp, X)] = _transversity_amps_deltaC(q2, deltaC7_dict[amp], '7', par)[(amp, X)]
    return ta

def transversity_amps_deltaC9(q2, deltaC9_dict, par):
    r"""A function returning a contribution to the transversity amplitudes in
    $\Lambda_b\to\Lambda\ell^+\ell^-$ coming from an effective transversity-dependent shift of
    the Wilson coefficient $C_7(\mu_b)$. This can be used to parametrize
    residual uncertainties due to subleading non-factorizable hadronic effects.

    The input dictionary `deltaC9_dict` should be of the form

    `{ 'perp0': deltaC9_perp0, 'para0': deltaC9_para0, 'perp1': deltaC9_perp1, 'para1': deltaC9_para1}`
    """
    ta = {}
    for amp in ['perp0', 'para0', 'perp1', 'para1']:
        for X in ['L', 'R']:
            ta[(amp, X)] = _transversity_amps_deltaC(q2, deltaC9_dict[amp], 'v', par)[(amp, X)]
    return ta

# One possibility is to parametrize the effective shift in C7 or C9 as a simple
# polynomial in q2.

def transversity_amps_deltaC7_polynomial(q2, par):
    deltaC7_dict = {}
    for amp in ['perp0', 'para0', 'perp1', 'para1']:
        deltaC7_dict[amp]  = ( par['Lambdab->Lambda deltaC7 a_' + amp + ' Re']
                             + par['Lambdab->Lambda deltaC7 b_' + amp + ' Re'] *q2
                             + 1j*par['Lambdab->Lambda deltaC7 a_' + amp + ' Im']
                             + 1j*par['Lambdab->Lambda deltaC7 b_' + amp + ' Im'] *q2)
    return transversity_amps_deltaC7(q2, deltaC7_dict, par)

# a constant shift, e.g. for high q^2
def transversity_amps_deltaC9_constant(q2, par):
    deltaC9_dict = {}
    for amp in ['perp0', 'para0', 'perp1', 'para1']:
        deltaC9_dict[amp]  = ( par['Lambdab->Lambda deltaC9 c_' + amp + ' Re']
                             + 1j*par['Lambdab->Lambda deltaC9 c_' + amp + ' Im'])
    return transversity_amps_deltaC9(q2, deltaC9_dict, par)


def fct_deltaC7_polynomial(wc_obj, par_dict, q2, cp_conjugate):
    par = par_dict.copy()
    if cp_conjugate:
        par = conjugate_par(par)
    return transversity_amps_deltaC7_polynomial(q2, par)


def fct_deltaC9_constant(wc_obj, par_dict, q2, cp_conjugate):
    par = par_dict.copy()
    if cp_conjugate:
        par = conjugate_par(par)
    return transversity_amps_deltaC9_constant(q2, par_dict)


# AuxiliaryQuantity & Implementatation: subleading effects at LOW q^2

quantity = 'Lambdab->Lambdall subleading effects at low q2'
a = AuxiliaryQuantity(name=quantity,
                      arguments=['q2', 'cp_conjugate'])
a.description = (r'Contribution to $\Lambda_b\to \Lambda \ell^+\ell^-$ transversity amplitudes from'
                 r' subleading hadronic effects (i.e. all effects not included'
                 r' elsewhere) at $q^2$ below the charmonium resonances')

# Implementation: C7-polynomial
iname = 'Lambdab->Lambdall deltaC7 polynomial'
i = Implementation(name=iname, quantity=quantity,
               function=fct_deltaC7_polynomial)
i.set_description(r"Effective shift in the Wilson coefficient $C_7(\mu_b)$"
                  r" as a first-order polynomial in $q^2$.")

# AuxiliaryQuantity & Implementatation: subleading effects at HIGH q^2

quantity = 'Lambdab->Lambdall subleading effects at high q2'
a = AuxiliaryQuantity(name=quantity, arguments=['q2', 'cp_conjugate'])
a.description = ('Contribution to $\Lambda_b\to \Lambda \ell^+\ell^-$ transversity amplitudes from'
                ' subleading hadronic effects (i.e. all effects not included'
                r'elsewhere) at $q^2$ above the charmonium resonances')

# Implementation: C9 constant shift
iname = 'Lambdab->Lambdall deltaC9 shift'
i = Implementation(name=iname, quantity=quantity,
               function=fct_deltaC9_constant)
i.set_description(r"Effective constant shift in the Wilson coefficient $C_9(\mu_b)$.")
