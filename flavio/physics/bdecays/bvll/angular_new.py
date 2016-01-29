from flavio.physics.bdecays.common import lambda_K
from math import sqrt, pi
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict, get_wceff
from flavio.physics.running import running
from flavio.physics.bdecays.formfactors import FormFactorParametrization as FF
from flavio.config import config
from flavio.physics.bdecays.common import lambda_K, beta_l, meson_quark, meson_ff
from flavio.physics.bdecays import matrixelements, angular
from flavio.physics import ckm

def prefactor(q2, par, B, V, lep):
    GF = par['Gmu']
    ml = par[('mass',lep)]
    scale = config['bdecays']['scale_bvll']
    alphaem = running.get_alpha(par, scale)['alpha_e']
    di_dj = meson_quark[(B,V)]
    xi_t = ckm.xi('t',di_dj)(par)
    if q2 <= 4*ml**2:
        return 0
    return 4*GF/sqrt(2)*xi_t*alphaem/(4*pi)

def get_ff(q2, par, B, V):
    return FF.parametrizations['bsz3'].get_ff(meson_ff[(B,V)], q2, par)

def get_angularcoeff(q2, wc_obj, par, B, V, lep):
    scale = config['bdecays']['scale_bvll']
    wc_tot = wctot_dict(wc_obj, meson_quark[(B,V)]+lep+lep, scale, par)
    wc = get_wceff(q2, wc_tot, par, B, V, lep, scale)
    ml = par[('mass',lep)]
    mB = par[('mass',B)]
    mV = par[('mass',V)]
    mb = running.get_mb(par, scale)
    N = prefactor(q2, par, B, V, lep)
    ff = get_ff(q2, par, B, V)
    h = angular.helicity_amps_v(q2, mB, mV, mb, 0, ml, ml, ff, wc, N)
    J = angular.angularcoeffs_general_v(h, q2, mB, mV, mb, 0, ml, ml)
    return J
