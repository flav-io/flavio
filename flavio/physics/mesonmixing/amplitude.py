"""Functions for the dispersive and absorptive  parts of the meson-antimeson mixing amplitude"""
# Note that the 12-amplitudes are defined as antimeson->meson.
# Thus, for Bq and K, the relevant quark-level transition is
# b-qbar->q-bbar and s-dbar->d-sbar, but for D it is u-cbar->c-ubar
# (and not c-ubar->u-cbar!)

from math import log,pi,sqrt
import flavio
from flavio.physics.mesonmixing.wilsoncoefficient import cvll_d
from flavio.physics.mesonmixing.common import meson_quark, bag_msbar2rgi
from flavio.config import config
from flavio.physics.running import running
from flavio.physics import ckm

def matrixelements(par, meson):
    r"""Returns a dictionary with the values of the matrix elements of the
    $\Delta F=2$ operators."""
    mM = par['m_'+meson]
    fM = par['f_'+meson]
    BM = lambda i: par['bag_' + meson + '_' + str(i)]
    qi_qj = meson_quark[meson]
    scale = config['renormalization scale'][meson + ' mixing']
    mq1 = running.get_mq(qi_qj[0], par, scale)
    mq2 = running.get_mq(qi_qj[1], par, scale)
    r = (mM/(mq1+mq2))**2
    me = {}
    me['CVLL'] =  mM*fM**2*(1/3.)*BM(1)
    me['CSLR'] =  mM*fM**2*(1/4.)*BM(4)*r
    me['CVRR'] = me['CVLL']
    me['CVLR'] = -mM*fM**2*(1/6.)*BM(5)*r
    me['CSLL'] = -mM*fM**2*(5/24.)*BM(2)*r
    me['CSRR'] = me['CSLL']
    me['CTLL'] = mM*fM**2*(1/2.)*r*(5*BM(2)/3.-2*BM(3)/3.)
    me['CTRR'] = me['CTLL']
    return me

def M12_d_SM(par, meson):
    r"""Standard model contribution to the mixing amplitude $M_{12}$ of
    meson $K^0$, $B^0$, or $B_s$.

    Defined as
    $$M_{12}^M = \frac{\langle M | \mathcal H_{\mathrm{eff}}|\bar M\rangle}{2M_M}$$
    (Attention: $M_{12}$ is sometimes defined with the meson and anti-meson
    switched in the above definition, leading to a complex conjugation.)
    """
    me = matrixelements(par, meson)
    scale = config['renormalization scale'][meson + ' mixing']
    alpha_s = running.get_alpha(par, scale)['alpha_s']
    me_rgi = me['CVLL'] * bag_msbar2rgi(alpha_s, meson)
    C_tt, C_cc, C_ct = cvll_d(par, meson)
    eta_tt = par['eta_tt_'+meson]
    M12  = - (eta_tt*C_tt) * me_rgi
    # charm contribution only needed for K mixing! Negligible for B and Bs.
    if meson == 'K0':
        eta_cc = par['eta_cc_'+meson]
        eta_ct = par['eta_ct_'+meson]
        M12 = M12 - (eta_cc*C_cc + eta_ct*C_ct) * me_rgi
    return M12


def M12(par, wc, meson):
    r"""Mixing amplitude $M_{12}$ of meson $K^0$, $B^0$, or $B_s$.

    Defined as
    $$M_{12}^M = \frac{\langle M | \mathcal H_{\mathrm{eff}}|\bar M\rangle}{2M_M}$$
    (Attention: $M_{12}$ is sometimes defined with the meson and anti-meson
    switched in the above definition, leading to a complex conjugation.)
    """
    me = matrixelements(par, meson)
    # new physics contributions to the mixing amplitude
    # the minus sign below stems from the fact that H_eff = -C_i O_i
    di_dj = meson_quark[meson]
    contribution_np = 0
    for me_name, me_value in me.items():
        wc_name = '{}_{}'.format(me_name, 2 * di_dj)
        contribution_np += -wc.get(wc_name, 0) * me_value
    # SM contribution
    if meson == 'D0':
        contribution_sm = M12_u_SM(par)
    else: # for B0, Bs, K0
        contribution_sm = M12_d_SM(par, meson)
    # new physics + SM
    return contribution_np + contribution_sm

def G12_d_SM(par, meson):
    flavio.citations.register("Beneke:2003az")
    di_dj = meson_quark[meson]
    xi_t = ckm.xi('t',di_dj)(par)
    xi_u = ckm.xi('u',di_dj)(par)
    c = par['Gamma12_'+meson+'_c']
    a = par['Gamma12_'+meson+'_a']
    M12 = M12_d_SM(par, meson)
    return M12*( c + a * xi_u/xi_t )*1e-4

def G12_d(par, wc, meson):
    #TODO at the moment NP contributions to Gamma_12 are ignored!
    return G12_d_SM(par, meson)

_ps = 1.519267515435317e+12 # picosecond in 1/GeV

def G12_u_SM(par):
    xi_b = ckm.xi('b', 'uc')(par)
    xi_s = ckm.xi('s', 'uc')(par)
    a_bb = par['Gamma12_D a_bb']
    a_bs = par['Gamma12_D a_bs']
    a_ss = par['Gamma12_D a_ss']
    return (a_ss * xi_s**2 + a_bs * xi_b*xi_s + a_bb * xi_b**2)/_ps

def M12_u_SM(par):
    xi_b = ckm.xi('b', 'uc')(par)
    xi_s = ckm.xi('s', 'uc')(par)
    a_bb = par['M12_D a_bb']
    a_bs = par['M12_D a_bs']
    a_ss = par['M12_D a_ss']
    return (a_ss * xi_s**2 + a_bs * xi_b*xi_s + a_bb * xi_b**2)/_ps

def G12_u(par, wc):
    #TODO at the moment NP contributions to Gamma_12 are ignored!
    return G12_u_SM(par)
