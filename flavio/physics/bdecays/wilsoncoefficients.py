from math import sqrt,pi,log
import pkgutil
import numpy as np
from io import StringIO
import scipy.interpolate
from flavio.physics.running import running
from flavio.physics import ckm
from flavio.physics.common import add_dict
from flavio.config import config
from flavio.physics.bdecays.common import meson_quark
from flavio.physics.bdecays import matrixelements

# SM Wilson coefficients at 120 GeV in the basis
# [ C_1, C_2, C_3, C_4, C_5, C_6,
# C_7^eff, C_8^eff,
# C_9, C_10,
# C_3^Q, C_4^Q, C_5^Q, C_6^Q,
# Cb ]
# where all operators are defined as in hep-ph/0512066 *except*
# C_9,10, which are defined with an additional alpha/4pi prefactor.
_wcsm_120 = np.zeros(34)
_wcsm_120[:15] = np.array([  1.99030910e-01,   1.00285703e+00,  -4.17672471e-04,
         2.00964137e-03,   5.20961618e-05,   9.65703651e-05,
        -1.98510105e-01,  -1.09453204e-01,   1.52918563e+00,
        -4.06926405e+00,   6.15944332e-03,   0.00000000e+00,
        -1.12876870e-03,   0.00000000e+00,  -3.24099235e-03])

def wctot_dict(wc_obj, sector, scale, par):
    r"""Get a dictionary with the total (SM + new physics) values  of the
    $\Delta F=1$ Wilson coefficients at a given scale, given a
    WilsonCoefficients instance."""
    wc_np_dict = wc_obj.get_wc(sector, scale, par)
    wc_sm = running.get_wilson(par, _wcsm_120, wc_obj.rge_derivative[sector], 120., scale)
    wc_labels = wc_obj.coefficients[sector]
    wc_sm_dict =  dict(zip(wc_labels, wc_sm))
    return add_dict((wc_np_dict, wc_sm_dict))

def get_wceff(q2, wc, par, B, M, lep, scale):
    """Get a dictionary with the effective $\Delta F=1$ Wilson coefficients
    in the convention appropriate for the generalized angular distributions.
    """
    xi_u = ckm.xi('u',meson_quark[(B,M)])(par)
    xi_t = ckm.xi('t',meson_quark[(B,M)])(par)
    qiqj=meson_quark[(B,M)]
    Yq2 = matrixelements.Y(q2, wc, par, scale, qiqj) + (xi_u/xi_t)*matrixelements.Yu(q2, wc, par, scale, qiqj)
        #   b) NNLO Q1,2
    delta_C7 = matrixelements.delta_C7(par=par, wc=wc, q2=q2, scale=scale, qiqj=qiqj)
    delta_C9 = matrixelements.delta_C9(par=par, wc=wc, q2=q2, scale=scale, qiqj=qiqj)
    mb = running.get_mb(par, scale)
    ll = lep + lep
    c = {}
    c['7']  = wc['C7eff_'+qiqj]      + delta_C7
    c['7p'] = wc['C7effp_'+qiqj]
    c['v']  = wc['C9_'+qiqj+ll]      + delta_C9 + Yq2
    c['vp'] = wc['C9p_'+qiqj+ll]
    c['a']  = wc['C10_'+qiqj+ll]
    c['ap'] = wc['C10p_'+qiqj+ll]
    c['s']  = mb * wc['CS_'+qiqj+ll]
    c['sp'] = mb * wc['CSp_'+qiqj+ll]
    c['p']  = mb * wc['CP_'+qiqj+ll]
    c['pp'] = mb * wc['CPp_'+qiqj+ll]
    c['t']  = 0
    c['tp'] = 0
    return c


def get_wceff_fccc(q2, wc_obj, par, B, P, lep):
    """Get a dictionary with the effective $b\to(c,u)$ Wilson coefficients
    in the convention appropriate for the generalized angular distributions.
    """
    scale = config['bdecays']['scale_bplnu']
    bqlnu = meson_quark[(B,P)] + lep + 'nu'
    wc = wc_obj.get_wc(bqlnu, scale, par)
    mb = running.get_mb(par, scale)
    c = {}
    c['7']  = 0
    c['7p'] = 0
    c['v']  = (1 + wc['CV_'+bqlnu])/2.
    c['vp'] = (1 + wc['CVp_'+bqlnu])/2.
    c['a']  = -wc['CV_'+bqlnu]/2.
    c['ap'] = -wc['CVp_'+bqlnu]/2.
    c['s']  = 1/2 * mb * wc['CS_'+bqlnu]/2.
    c['sp'] = 1/2 * mb * wc['CSp_'+bqlnu]/2.
    c['p']  = -1/2 * mb * wc['CS_'+bqlnu]/2.
    c['pp'] = -1/2 * mb * wc['CSp_'+bqlnu]/2.
    c['t']  = wc['CT_'+bqlnu]
    c['tp'] = 0
    return c
