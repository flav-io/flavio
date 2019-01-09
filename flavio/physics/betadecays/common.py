from flavio.physics.edms.common import proton_charges
from flavio.config import config
from flavio.physics.bdecays.wilsoncoefficients import get_wceff_fccc_std


def wc_eff(par, wc_obj, scale, nu):
    r"""Lee-Yang effective couplings.

    See eqS. (2), (9) of arXiv:1803.08732."""
    # wilson coefficients
    scale = config['renormalization scale']['betadecay']
    wc = get_wceff_fccc_std(wc_obj, par, 'du', 'e', nu, None, scale, nf=3)
    # proton charges
    g = proton_charges(par, scale)
    gV = g['gV_u-d']
    gA = g['gA_u-d']
    gS = g['gS_u-d']
    gP = g['gP_u-d']
    gT = g['gT_u-d']
    # effective couplings
    C = {}
    C['V'] =  2 * gV * (wc['VL'] + wc['VR'])
    C['A'] = -2 * gA * (wc['VL'] - wc['VR'])
    C['S'] =      gS * (wc['SL'] + wc['SR'])
    C['P'] =      gP * (wc['SL'] - wc['SR'])
    C['T'] =  2 * gT * (wc['T'])
    return C
