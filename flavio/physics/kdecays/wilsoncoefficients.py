r"""Functions for SM Wilson coefficients in kaon decays."""

import scipy.interpolate
import flavio
from flavio.physics import ckm


# Table 1 of 1507.06345: Wilson coefficients at 1.3 GeV
# for alpha_s = [0.1179, 0.1185, 0.1191]
_yz = [[-0.4036, -0.4092, -0.4150],
[1.2084, 1.2120, 1.2157],
[0.0275, 0.0280, 0.0285],
[-0.0555, -0.0563, -0.0571],
[0.0054, 0.0052, 0.0050],
[-0.0849, -0.0867, -0.0887],
[-0.0404, -0.0403, -0.0402],
[0.1207, 0.1234, 0.1261],
[-1.3936, -1.3981, -1.4027],
[0.4997, 0.5071, 0.5146]]
_yz_rows = ["z1", "z2", "y3", "y4", "y5", "y6", "y7/al", "y8/al", "y9/al", "y10/al",]
# inter- & extrapolating alpha_s dependence
wcsm = scipy.interpolate.interp1d([0.1179, 0.1185, 0.1191], _yz, fill_value="extrapolate")


def wilsoncoefficients_sm_fourquark(par, scale):
    r"""Return the $\Delta S=1$ Wilson coefficients of four-quark operators
    in the SM at the scale `scale`.

    Currently only implemented for `scale=1.3`."""
    if scale != 1.3:
        raise ValueError("Wilson coefficients only implemented for scale=1.3")
    flavio.citations.register("Buras:2015yba")
    wcarr = wcsm(par['alpha_s'])
    wc_dict = dict(zip(["z1", "z2", "y3", "y4", "y5", "y6",
                        "y7/al", "y8/al", "y9/al", "y10/al",], wcarr))
    for k in ['y7', 'y8', 'y9', 'y10']:
        wc_dict[k] = wc_dict.pop('{}/al'.format(k)) / 128
    return wc_dict


def wilsoncoefficients_sm_sl(par, scale):
    r"""Return the $\Delta S=1$ Wilson coefficients of semi-leptonic operators
    in the SM at the scale `scale`.

    Currently only $C_{10}$ (top and charm contributions) is implemented."""
    wc_dict = {}
    # fold in approximate m_t-dependence of C_10 (see eq. 4 of arXiv:1311.0903)
    flavio.citations.register("Bobeth:2013uxa")
    wc_dict['C10_t'] = -4.10  * (par['m_t']/173.1)**1.53
    Vus = abs(ckm.get_ckm(par)[0, 1])
    Pc = 0.115 # +-0.011, arXiv:hep-ph/0605203
    flavio.citations.register("Gorbahn:2006bm")
    wc_dict['C10_c'] = -Pc / par['s2w'] * Vus**4
    return wc_dict
