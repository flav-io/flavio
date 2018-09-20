r"""Functions to parametrize subleading hadronic effects in $B\to V\ell^+\ell^-$
decays."""


import flavio
from flavio.classes import AuxiliaryQuantity, Implementation
from flavio.physics.common import conjugate_par
from flavio.config import config


class HelicityAmpsDeltaC(object):
    r"""Base Class for functions returning the shift of helcity amps for a
    shift in Wilson coefficients. Facilitates caching.

    This can be used to parametrize residual uncertainties due to subleading
    non-factorizable hadronic effects."""

    def __init__(self, B, V, par, q2):
        """Initialize the class and cache results needed more often."""
        self.B = B
        self.V = V
        self.q2 = q2
        self.par = par
        self.prefactor = flavio.physics.bdecays.bvll.amplitudes.prefactor(None, self.par, B, V)
        self.scale = config['renormalization scale']['bvll']
        self.mb = flavio.physics.running.running.get_mb(par, self.scale)
        self.mB = par['m_'+B]
        self.mV = par['m_'+V]
        self.ff = flavio.physics.bdecays.bvll.amplitudes.get_ff(self.q2, self.par, self.B, self.V)

    def ha_deltaC(self, deltaC, C_name):
        wc  = {'7': 0, '7p': 0, 'v': 0, 'a': 0, 's': 0, 'p': 0,
               't': 0,'vp': 0, 'ap': 0, 'sp': 0, 'pp': 0, 'tp': 0, }
        wc[C_name] = deltaC
        return flavio.physics.bdecays.angular.helicity_amps_v(self.q2,
                    self.mB, self.mV, self.mb, 0,
                    0, 0,  # ml=0 as this only affects scalar contributions
                    self.ff, wc, self.prefactor)


class HelicityAmpsDeltaC_77p_polynomial(HelicityAmpsDeltaC):
    r"""Helicity amps from an effective shift in the Wilson coefficients
    $C_7$ (for the 0 and $-$ amplitudes) and $C_7'$ (for the $+$) amplitude
    as a polynomial in $q^2$."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self):
        par = self.par
        B = self.B
        V = self.V
        q2 = self.q2
        deltaC7_0   =( par[B+'->'+V+' deltaC7 a_0 Re']  + par[B+'->'+V+' deltaC7 b_0 Re'] *q2
                 +1j*( par[B+'->'+V+' deltaC7 a_0 Im']  + par[B+'->'+V+' deltaC7 b_0 Im'] *q2 ))
        deltaC7p_pl  =( par[B+'->'+V+' deltaC7p a_+ Re']  + par[B+'->'+V+' deltaC7p b_+ Re'] *q2
                 +1j*( par[B+'->'+V+' deltaC7p a_+ Im']  + par[B+'->'+V+' deltaC7p b_+ Im'] *q2 ))
        deltaC7_mi  =( par[B+'->'+V+' deltaC7 a_- Re']  + par[B+'->'+V+' deltaC7 b_- Re'] *q2
                 +1j*( par[B+'->'+V+' deltaC7 a_- Im']  + par[B+'->'+V+' deltaC7 b_- Im'] *q2 ))
        ha = {}
        ha['0', 'V'] = self.ha_deltaC(deltaC7_0, '7')['0', 'V']
        ha['pl', 'V'] = self.ha_deltaC(deltaC7p_pl, '7p')['pl', 'V']
        ha['mi', 'V'] = self.ha_deltaC(deltaC7_mi, '7')['mi', 'V']
        return ha

class HelicityAmpsDeltaC_9_shift(HelicityAmpsDeltaC):
    r"""Helicity amps from an effective $q^2$-independent shift in the Wilson
    coefficients."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self):
        par = self.par
        B = self.B
        V = self.V
        deltaC9_0   = par[B+'->'+V+' deltaC9 c_0 Re'] + 1j*par[B+'->'+V+' deltaC9 c_0 Im']
        deltaC9_pl   = par[B+'->'+V+' deltaC9 c_+ Re'] + 1j*par[B+'->'+V+' deltaC9 c_+ Im']
        deltaC9_mi   = par[B+'->'+V+' deltaC9 c_- Re'] + 1j*par[B+'->'+V+' deltaC9 c_- Im']
        ha = {}
        ha['0', 'V'] = self.ha_deltaC(deltaC9_0, '9')['0', 'V']
        ha['pl', 'V'] = self.ha_deltaC(deltaC9_pl, '9')['pl', 'V']
        ha['mi', 'V'] = self.ha_deltaC(deltaC9_mi, '9')['mi', 'V']
        return ha


def fct_deltaC7C7p_polynomial(B, V):
    def fct(wc_obj, par_dict, q2, cp_conjugate):
        par = par_dict.copy()
        return HelicityAmpsDeltaC_77p_polynomial(B, V, par, q2)()
    return fct


def fct_deltaC9_constant(B, V):
    def fct(wc_obj, par_dict, q2, cp_conjugate):
        par = par_dict.copy()
        return HelicityAmpsDeltaC_9_shift(B, V, par, q2)()
    return fct

# AuxiliaryQuantity & Implementation: subleading effects at LOW q^2

for had in [('B0','K*0'), ('B+','K*+'), ('Bs','phi'), ]:
    process = had[0] + '->' + had[1] + 'll' # e.g. B0->K*0mumu
    quantity = process + ' subleading effects at low q2'
    a = AuxiliaryQuantity(name=quantity, arguments=['q2', 'cp_conjugate'])
    a.description = ('Contribution to ' + process + ' helicity amplitudes from'
                    ' subleading hadronic effects (i.e. all effects not included'
                    r'elsewhere) at $q^2$ below the charmonium resonances')


    # Implementation: C7-C7'-polynomial
    iname = process + ' deltaC7, 7p polynomial'
    i = Implementation(name=iname, quantity=quantity,
                   function=fct_deltaC7C7p_polynomial(B=had[0], V=had[1]))
    i.set_description(r"Effective shift in the Wilson coefficient $C_7(\mu_b)$"
                      r" (in the $0$ and $-$ helicity amplitudes) and"
                      r" $C_7'(\mu_b)$ (in the $+$ helicity amplitude)"
                      r" as a first-order polynomial in $q^2$.")


# AuxiliaryQuantity & Implementation: subleading effects at HIGH q^2

for had in [('B0','K*0'), ('B+','K*+'), ('Bs','phi'), ]:
    process = had[0] + '->' + had[1] + 'll' # e.g. B0->K*0mumu
    quantity = process + ' subleading effects at high q2'
    a = AuxiliaryQuantity(name=quantity, arguments=['q2', 'cp_conjugate'])
    a.description = ('Contribution to ' + process + ' helicity amplitudes from'
                    ' subleading hadronic effects (i.e. all effects not included'
                    r'elsewhere) at $q^2$ above the charmonium resonances')

    # Implementation: C9 constant shift
    iname = process + ' deltaC9 shift'
    i = Implementation(name=iname, quantity=quantity,
                   function=fct_deltaC9_constant(B=had[0], V=had[1]))
    i.set_description(r"Effective constant shift in the Wilson coefficient $C_9(\mu_b)$.")
