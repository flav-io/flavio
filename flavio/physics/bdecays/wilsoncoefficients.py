r"""Standard Model Wilson coefficients for $\Delta B=1$ transitions as well
as tree-level $B$ decays"""


from math import sqrt, log, pi
import numpy as np
import scipy.interpolate
from flavio.physics.running import running
from flavio.physics import ckm
from flavio.physics.common import add_dict
from flavio.physics.bdecays.common import meson_quark
from flavio.physics.bdecays import matrixelements
import flavio
import copy
import pkg_resources


# SM Wilson coefficients for n_f=5 in the basis
# [ C_1, C_2, C_3, C_4, C_5, C_6,
# C_7^eff, C_8^eff,
# C_9, C_10,
# C_3^Q, C_4^Q, C_5^Q, C_6^Q,
# Cb ]
# where all operators are defined as in hep-ph/0512066 *except*
# C_9,10, which are defined with an additional alpha/4pi prefactor.
scales = (2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5)
# data = np.array([C_low(s, 120, get_par(), nf=5) for s in scales]).T
data = np.load(pkg_resources.resource_filename('flavio.physics', 'data/wcsm/wc_sm_dB1_2_55.npy'))
wcsm_nf5 = scipy.interpolate.interp1d(scales, data)

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
    flavio.citations.register("Brod:2010hi")
    XtEW = Xt0 * (1 - 1.11508 + 1.12316*1.15338**(mt/165.)-0.179454*(mt/165)) - 1
    XtEW = XtEW * 0.00062392534457616328 # <- alpha_em/4pi at 120 GeV
    Xt = Xt0 + Xt1 + XtEW
    return -Xt/s2w


# names of SM DeltaF=1 Wilson coefficients needed for wctot_dict
fcnclabels = {}
_fcnc = ['bs', 'bd', 'sd', ]
_ll = ['ee', 'mumu', 'tautau']
for qq in _fcnc:
    for ll in _ll:
        fcnclabels[qq + ll] = ['C1_'+qq, 'C2_'+qq, # current-current
                               'C3_'+qq, 'C4_'+qq, 'C5_'+qq, 'C6_'+qq, # QCD penguins
                               'C7_'+qq, 'C8_'+qq, # dipoles
                               'C9_'+qq+ll, 'C10_'+qq+ll, # semi-leptonic
                               'C3Q_'+qq, 'C4Q_'+qq, 'C5Q_'+qq, 'C6Q_'+qq, 'Cb_'+qq, # EW penguins
                                # and everything with flipped chirality ...
                               'C1p_'+qq, 'C2p_'+qq,
                               'C3p_'+qq, 'C4p_'+qq, 'C5p_'+qq, 'C6p_'+qq,
                               'C7p_'+qq, 'C8p_'+qq,
                               'C9p_'+qq+ll, 'C10p_'+qq+ll,
                               'C3Qp_'+qq, 'C4Qp_'+qq, 'C5Qp_'+qq, 'C6Qp_'+qq, 'Cbp_'+qq,
                                # scalar and pseudoscalar
                               'CS_'+qq+ll, 'CP_'+qq+ll,
                               'CSp_'+qq+ll, 'CPp_'+qq+ll, ]


def wctot_dict(wc_obj, sector, scale, par, nf_out=5):
    r"""Get a dictionary with the total (SM + new physics) values  of the
    $\Delta F=1$ Wilson coefficients at a given scale, given a
    WilsonCoefficients instance."""
    wc_np_dict = wc_obj.get_wc(sector, scale, par, nf_out=nf_out)
    if nf_out == 5:
        wc_sm = wcsm_nf5(scale)
    else:
        raise NotImplementedError("DeltaF=1 Wilson coefficients only implemented for B physics")
    # fold in approximate m_t-dependence of C_10 (see eq. 4 of arXiv:1311.0903)
    flavio.citations.register("Bobeth:2013uxa")
    wc_sm[9] = wc_sm[9] * (par['m_t']/173.1)**1.53
    # go from the effective to the "non-effective" WCs for C7 and C8
    yi = np.array([0, 0, -1/3., -4/9., -20/3., -80/9.])
    zi = np.array([0, 0, 1, -1/6., 20, -10/3.])
    wc_sm[6] = wc_sm[6] - np.dot(yi, wc_sm[:6]) # c7 (not effective!)
    wc_sm[7] = wc_sm[7] - np.dot(zi, wc_sm[:6]) # c8 (not effective!)
    wc_labels = fcnclabels[sector]
    wc_sm_dict = dict(zip(wc_labels, wc_sm))
    # now here comes an ugly fix. If we have b->s transitions, we should take
    # into account the fact that C7' = C7*ms/mb, and the same for C8, which is
    # not completely negligible. To find out whether we have b->s, we look at
    # the "sector" string.
    if sector[:2] == 'bs':
        eps_s = running.get_ms(par, scale)/running.get_mb(par, scale)
        wc_sm_dict['C7p_bs'] = eps_s * wc_sm_dict['C7_bs']
        wc_sm_dict['C8p_bs'] = eps_s * wc_sm_dict['C8_bs']
    tot_dict = add_dict((wc_np_dict, wc_sm_dict))
    # add C7eff(p) and C8eff(p)
    tot_dict.update(get_C78eff(tot_dict, sector[:2]))
    return tot_dict

def get_C78eff(wc, qiqj):
    r"""Return the effective Wilson coefficients $C_{7,8}^\text{eff}$
    for sector `qiqj` (e.g. 'bs')."""
    yi = np.array([-1/3., -4/9., -20/3., -80/9.])
    zi = np.array([1, -1/6., 20, -10/3.])
    wceff = {}
    C36 = np.array([wc[k] for k in ['C3_'+qiqj, 'C4_'+qiqj, 'C5_'+qiqj, 'C6_'+qiqj]])
    C36p = np.array([wc[k] for k in ['C3p_'+qiqj, 'C4p_'+qiqj, 'C5p_'+qiqj, 'C6p_'+qiqj]])
    wceff['C7eff_' + qiqj] = wc['C7_' + qiqj] + np.dot(yi, C36)
    wceff['C8eff_' + qiqj] = wc['C8_' + qiqj] + np.dot(zi, C36)
    wceff['C7effp_' + qiqj] = wc['C7p_' + qiqj] + np.dot(yi, C36p)
    wceff['C8effp_' + qiqj] = wc['C8p_' + qiqj] + np.dot(zi, C36p)
    return wceff

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

def get_CVLSM(par, scale, nf):
    r"""Get the Wilson coefficient of the operator $C_V$ in $d_i\to d_j\ell\nu$
    in the SM including EW corrections."""
    if nf >= 4: # for B and D physics
        alpha_e = running.get_alpha(par, scale)['alpha_e']
        return 1 + alpha_e/pi * log(par['m_Z']/scale)
    else: # for K and pi physics
        # Marciano & Sirlin 1993
        return sqrt(1.0232)

def get_wceff_fccc(wc_obj, par, qiqj, lep, nu, mqi, scale, nf=5):
    r"""Get a dictionary with the $d_i\to d_j$ Wilson coefficients
    in the convention appropriate for the generalized angular distributions.
    """
    qqlnu = qiqj + lep + 'nu' + nu
    wc = wc_obj.get_wc(qqlnu, scale, par, nf_out=nf)
    if lep == nu:
        c_sm = get_CVLSM(par, scale, nf)
    else:
        c_sm = 0  # SM contribution only for neutrino flavor = lepton flavor
    c = {}
    c['7']  = 0
    c['7p'] = 0
    c['v']  = (c_sm + wc['CVL_'+qqlnu])/2.
    c['vp'] = wc['CVR_'+qqlnu]/2.
    c['a']  = -(c_sm + wc['CVL_'+qqlnu])/2.
    c['ap'] = -wc['CVR_'+qqlnu]/2.
    c['s']  = wc['CSR_'+qqlnu]/2.
    c['sp'] = wc['CSL_'+qqlnu]/2.
    c['p']  = -wc['CSR_'+qqlnu]/2.
    c['pp'] = -wc['CSL_'+qqlnu]/2.
    c['t']  = 0
    c['tp'] = wc['CT_'+qqlnu]
    return c


def get_wceff_fccc_std(wc_obj, par, qiqj, lep, nu, mqi, scale, nf=5):
    r"""Get a dictionary with the $d_i\to d_j$ Wilson coefficients
    in the flavio default convention.
    """
    qqlnu = qiqj + lep + 'nu' + nu
    wc = wc_obj.get_wc(qqlnu, scale, par, nf_out=nf)
    if lep == nu:
        c_sm = get_CVLSM(par, scale, nf)
    else:
        c_sm = 0  # SM contribution only for neutrino flavor = lepton flavor
    c = {}
    c['VL']  = c_sm + wc['CVL_'+qqlnu]
    c['VR'] = wc['CVR_'+qqlnu]
    c['SR']  = wc['CSR_'+qqlnu]
    c['SL'] = wc['CSL_'+qqlnu]
    c['T']  = wc['CT_'+qqlnu]
    return c
