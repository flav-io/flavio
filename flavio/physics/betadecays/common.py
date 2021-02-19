"""Common functions for beta decays."""

import flavio
from flavio.physics.edms.common import proton_charges
from flavio.physics.bdecays.wilsoncoefficients import get_wceff_fccc_std, get_CVLSM
from math import sqrt


def wc_eff(par, wc_obj, scale, nu):
    r"""Lee-Yang effective couplings.

    See eqS. (2), (9) of arXiv:1803.08732."""
    flavio.citations.register("Gonzalez-Alonso:2018omy")
    # wilson coefficients
    wc = get_wceff_fccc_std(wc_obj, par, 'du', 'e', nu, None, scale, nf=3)
    # proton charges
    g = proton_charges(par, scale)
    gV = g['gV_u-d']
    gA = g['gA_u-d']
    gS = g['gS_u-d']
    gP = g['gP_u-d']
    gT = g['gT_u-d']
    # radiative corrections
    # Note: CVLSM is the universal Marciano-Sirlin result that needs to be
    # divided out since it's already  contained in the Deltas
    CVLSM = get_CVLSM(par, scale, nf=3)
    DeltaRV = par['DeltaRV']
    DeltaRA = DeltaRV  # not needed for superallowed, for neutron difference absorbed in lambda
    rV = sqrt(1 + DeltaRV) / CVLSM
    rA = sqrt(1 + DeltaRA) / CVLSM
    # effective couplings
    # note that C_i' = C_i
    C = {}
    C['V'] = gV * (wc['VL'] * rV + wc['VR'])
    C['A'] = -gA * (wc['VL'] * rA - wc['VR'])
    C['S'] = gS * (wc['SL'] + wc['SR'])
    C['P'] = gP * (wc['SL'] - wc['SR'])
    C['T'] = 4 * gT * (wc['T'])
    return C
