from flavio.physics.bdecays.common import lambda_K
from math import sqrt, pi
from flavio.physics.bdecays.wilsoncoefficients import get_wceff
from flavio.physics.running import running
from flavio.physics.bdecays.formfactors import FormFactorParametrization as FF
from flavio.config import config
from flavio.physics.bdecays.common import lambda_K, beta_l, meson_quark, meson_ff
from flavio.physics.bdecays import matrixelements, angular
from flavio.physics import ckm

def prefactor(q2, par, B, P, lep):
    GF = par['Gmu']
    ml = par[('mass',lep)]
    scale = config['bdecays']['scale_bpll']
    alphaem = running.get_alpha(par, scale)['alpha_e']
    di_dj = meson_quark[(B,P)]
    xi_t = ckm.xi('t',di_dj)(par)
    if q2 <= 4*ml**2:
        return 0
    return 4*GF/sqrt(2)*xi_t*alphaem/(4*pi)

def get_ff(q2, par, B, P):
    return FF.parametrizations['btop_lattice'].get_ff(meson_ff[(B,P)], q2, par)

def get_angularcoeff(q2, wc_obj, par, B, P, lep):
    scale = config['bdecays']['scale_bpll']
    wc = get_wceff(q2, wc_obj, par, B, P, lep, scale)
    ml = par[('mass',lep)]
    mB = par[('mass',B)]
    mP = par[('mass',P)]
    mb = running.get_mb(par, scale)
    N = prefactor(q2, par, B, P, lep)
    ff = get_ff(q2, par, B, P)
    h = angular.helicity_amps_p(q2, mB, mP, mb, 0, ml, ml, ff, wc, N)
    J = angular.angularcoeffs_general_p(h, q2, mB, mP, mb, 0, ml, ml)
    return J
