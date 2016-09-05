r"""Standard Model Wilson coefficients for $\Delta B=1$ transitions as well
as tree-level $B$ decays"""

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
import flavio

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
        -1.98501275e-01,  -1.09453204e-01,   1.52918563e+00,
        -4.06926405e+00,   6.15944332e-03,   0.00000000e+00,
        -1.12876870e-03,   0.00000000e+00,  -3.24099235e-03])

# di->djnunu Wilson coefficient
def CL_SM(par):
    r"""SM Wilson coefficient for $d_i\to d_j\nu\bar\nu$ transitions.

    This is implemented as an approximate formula as a function of the top
    mass."""
    # EW NLO corrections arXiv:1009.0947
    scale = 120. # <- result has very little sensitivity to high matching scale
    mt = flavio.physics.running.running.get_mt(par, scale)
    s2w = par['s2w']
    Xt0_165 = 1.50546 # LO result for mt=165, scale=120
    Xt0 = Xt0_165 * (1 + 1.14064 * (mt/165. - 1)) # LO
    Xt1 = Xt0_165 * (-0.031435 - 0.139303 * (mt/165. - 1)) # QCD NLO
    # (4.3), (4.4) of 1009.0947: NLO EW
    XtEW = Xt0 * (1 - 1.11508 + 1.12316*1.15338**(mt/165.)-0.179454*(mt/165)) - 1
    XtEW = XtEW * 0.00062392534457616328 # <- alpha_em/4pi at 120 GeV
    Xt = Xt0 + Xt1 + XtEW
    return -Xt/s2w


def wctot_dict(wc_obj, sector, scale, par, nf_out=None):
    r"""Get a dictionary with the total (SM + new physics) values  of the
    $\Delta F=1$ Wilson coefficients at a given scale, given a
    WilsonCoefficients instance."""
    wc_np_dict = wc_obj.get_wc(sector, scale, par, nf_out=nf_out)
    wcsm_120 = _wcsm_120.copy()
    wc_sm = running.get_wilson(par, wcsm_120, wc_obj.rge_derivative[sector], 120., scale, nf_out=nf_out)
    # now here comes an ugly fix. If we have b->s transitions, we should take
    # into account the fact that C7' = C7*ms/mb, and the same for C8, which is
    # not completely negligible. To find out whether we have b->s, we look at
    # the "sector" string.
    if sector[:2] == 'bs':
        # go from the effective to the "non-effective" WCs
        yi = np.array([0, 0, -1/3., -4/9., -20/3., -80/9.])
        zi = np.array([0, 0, 1, -1/6., 20, -10/3.])
        c7 = wc_sm[6] - np.dot(yi, wc_sm[:6]) # c7 (not effective!)
        c8 = wc_sm[7] - np.dot(zi, wc_sm[:6]) # c8 (not effective!)
        eps_s = running.get_ms(par, scale)/running.get_mb(par, scale)
        c7p = eps_s * c7
        c8p = eps_s * c8
        # go back to the effective WCs
        wc_sm[21] = c7p + np.dot(yi, wc_sm[15:21]) # c7p_eff
        wc_sm[22] = c7p + np.dot(zi, wc_sm[15:21]) # c8p_eff
    wc_labels = wc_obj.coefficients[sector]
    wc_sm_dict =  dict(zip(wc_labels, wc_sm))
    return add_dict((wc_np_dict, wc_sm_dict))

def get_wceff(q2, wc, par, B, M, lep, scale):
    r"""Get a dictionary with the effective $\Delta F=1$ Wilson coefficients
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

def get_wceff_lfv(q2, wc, par, B, M, l1, l2, scale):
    r"""Get a dictionary with the effective $\Delta F=1$ Wilson coefficients
    with lepton flavour violation
    in the convention appropriate for the generalized angular distributions.
    """
    mb = running.get_mb(par, scale)
    qiqj=meson_quark[(B,M)]
    c = {}
    c['7']  = 0
    c['7p'] = 0
    c['v']  = wc['C9_'+qiqj+l1+l2]
    c['vp'] = wc['C9p_'+qiqj+l1+l2]
    c['a']  = wc['C10_'+qiqj+l1+l2]
    c['ap'] = wc['C10p_'+qiqj+l1+l2]
    c['s']  = mb * wc['CS_'+qiqj+l1+l2]
    c['sp'] = mb * wc['CSp_'+qiqj+l1+l2]
    c['p']  = mb * wc['CP_'+qiqj+l1+l2]
    c['pp'] = mb * wc['CPp_'+qiqj+l1+l2]
    c['t']  = 0
    c['tp'] = 0
    return c


def get_wceff_nunu(q2, wc, par, B, M, nu1, nu2, scale):
    r"""Get a dictionary with the effective $\Delta F=1$ Wilson coefficients
    with neutrinos in the final state
    in the convention appropriate for the generalized angular distributions.
    """
    qiqj=meson_quark[(B,M)]
    c = {}
    c['7']  = 0
    c['7p'] = 0
    c['v']  =  wc['CL_'+qiqj+nu1+nu2]
    c['vp'] =  wc['CR_'+qiqj+nu1+nu2]
    c['a']  = -wc['CL_'+qiqj+nu1+nu2]
    c['ap'] = -wc['CR_'+qiqj+nu1+nu2]
    c['s']  = 0
    c['sp'] = 0
    c['p']  = 0
    c['pp'] = 0
    c['t']  = 0
    c['tp'] = 0
    return c

def get_CVSM(par, scale, nf):
    r"""Get the Wilson coefficient of the operator $C_V$ in $d_i\to d_j\ell\nu$
    in the SM including EW corrections."""
    if nf >= 4: # for B and D physics
        alpha_e = running.get_alpha(par, scale)['alpha_e']
        return 1 + alpha_e/pi * log(par['m_Z']/scale)
    else: # for K and pi physics
        # Marciano & Sirlin 1993
        return sqrt(1.0232)

def get_wceff_fccc(wc_obj, par, qiqj, lep, mqi, scale, nf=5):
    r"""Get a dictionary with the $d_i\to d_j$ Wilson coefficients
    in the convention appropriate for the generalized angular distributions.
    """
    qqlnu = qiqj + lep + 'nu'
    wc = wc_obj.get_wc(qqlnu, scale, par)
    c_sm = get_CVSM(par, scale, nf)
    c = {}
    c['7']  = 0
    c['7p'] = 0
    c['v']  = (c_sm + wc['CV_'+qqlnu])/2.
    c['vp'] = wc['CVp_'+qqlnu]/2.
    c['a']  = -(c_sm + wc['CV_'+qqlnu])/2.
    c['ap'] = -wc['CVp_'+qqlnu]/2.
    c['s']  = 1/2 * mqi * wc['CS_'+qqlnu]/2.
    c['sp'] = 1/2 * mqi * wc['CSp_'+qqlnu]/2.
    c['p']  = -1/2 * mqi * wc['CS_'+qqlnu]/2.
    c['pp'] = -1/2 * mqi * wc['CSp_'+qqlnu]/2.
    c['t']  = wc['CT_'+qqlnu]
    c['tp'] = 0
    return c
