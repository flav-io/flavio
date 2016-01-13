from math import sqrt,pi
from flavio.physics.bdecays.common import lambda_K, meson_quark, meson_ff
from flavio.physics import ckm
from flavio.physics.bdecays.formfactors import FormFactorParametrization as FF

r"""Functions for exclusive $B\to V\ell\nu$ decays."""

def helicity_amps(q2, par, B, V, lep):
    GF = par['Gmu']
    aem = par['alphaem']
    ml = par[('mass',lep)]
    mB = par[('mass',B)]
    mV = par[('mass',V)]
    tauB = par[('lifetime',B)]
    la = lambda_K(mB**2,q2,mV**2)
    sla = sqrt(la)
    N = sqrt(GF**2/(192. * pi**3 * mB**3) * sla)
    ff = FF.parametrizations['bsz3'].get_ff(meson_ff[(B,V)], q2, par)
    ha = {}
    ha['+'] = -N *((mB + mV)*ff['A1'] - (sla/(mB + mV)) * ff['V'])
    ha['-'] = -N *((mB + mV)*ff['A1'] + (sla/(mB + mV)) * ff['V'])
    ha['0'] = -N *(1/sqrt(q2) * 8 * mB * mV * ff['A12'])
    return ha

def dGamma(ha, q2, Vub):
    amps2 =  (abs(ha['+'])**2 + abs(ha['-'])**2 + abs(ha['0'])**2)
    return abs(Vub)**2 * q2 * amps2

def dBR(q2, par, B, V, lep):
    ha = helicity_amps(q2, par, B, V, lep)
    Vub = abs(ckm.get_ckm(par)[0,2])
    tauB = par[('lifetime',B)]
    return tauB * dGamma(ha, q2, Vub)
