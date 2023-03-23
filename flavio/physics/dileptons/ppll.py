from flavio.math.integrate import nintegrate
from flavio.physics.zdecays.smeftew import gV_SM, gA_SM, _QN
from flavio.physics.common import add_dict
from flavio.classes import Observable, Prediction
from flavio.physics import ckm as ckm_flavio
from . import partondist
import wilson
from numpy import pi, sqrt
import numpy as np
from flavio.config import config

def F_qqll_SM(q, Xq, l, Xl, s, par):
    r"""SM $Z$ and $\gamma$ contribution to the $\bar q q\to \ell^+\ell^-$
    amplitude."""
    # parameters
    Ql = -1
    mZ = par['m_Z']
    GammaZ = 1 / par['tau_Z']
    s2w = par['s2w']
    aEW = par['alpha_e']
    e2 = 4 * pi * aEW
    g2=e2/s2w * 0.9608 # correction factor from running g^2 to 1TeV
    gp2=e2/(1-s2w) * 1.0272 # correction factor from running g'^2 to 1TeV
    g_cw = sqrt(g2+gp2) # g over cw
    s2w = gp2/(g2+gp2)
    par = par.copy()
    par['s2w'] = s2w
    # SM couplings
    Qq = _QN[q]['Q']
    gVq = g_cw * gV_SM(q, par)
    gVl = g_cw * gV_SM(l, par)
    gAq = g_cw * gA_SM(q, par)
    gAl = g_cw * gA_SM(l, par)
    if Xq == 'L':
        gZq = gVq + gAq
    elif Xq == 'R':
        gZq = gVq - gAq
    if Xl == 'L':
        gZl = gVl + gAl
    elif Xl == 'R':
        gZl = gVl - gAl
    # SM contribution
    return g2*s2w * Qq * Ql / s + gZq * gZl / (s - mZ**2 + 1j * mZ * GammaZ)


def wceff_qqll_sm(s, par):
    E = np.einsum('ij,kl->ijkl', np.eye(3), np.eye(3))
    wc = {}
    for Xl in 'LR':
        for Xq in 'LR':
            for q in 'ud':
                wc['CV{}{}_e{}'.format(Xl, Xq, q)] = F_qqll_SM(q, Xq, 'e', Xl, s, par) * E
    wc['CSRL_ed'] = wc['CSRR_ed'] = wc['CTRR_ed'] = np.zeros((3, 3, 3, 3))
    wc['CSRL_eu'] = wc['CSRR_eu'] = wc['CTRR_eu'] = np.zeros((3, 3, 3, 3))
    return wc


def wceff_qqll_np(wc_obj, par, scale):
    r"""Returns Wilson coefficients of the effective Lagrangian
    $$\mathcal{L} = \sum_{q=u,d} C_{eq}^{\Gamma_1\Gamma_2}
    (\bar e_i\Gamma_1 e_j)(\bar q_k\Gamma_2 q_l)$$
    as a dictionary of arrays with shape (3, 3, 3, 3) corresponding to ijkl.
    """
    # get the dictionary
    wcxf_dict = wc_obj.get_wcxf(sector='dB=dL=0', scale=scale, par=par,
                                eft='SMEFT', basis='Warsaw up').dict
    # go to redundant basis, C has all the smeft coefficients
    C = wilson.util.smeftutil.wcxf2arrays_symmetrized(wcxf_dict)

    wc = {}
    ckm = ckm_flavio.get_ckm(par)
    # match to notation used in the observable
    wc['CVLL_eu'] = C['lq1'] - C['lq3']
    wc['CVLL_ed'] = np.einsum('ijmn,mk,nl->ijkl',C['lq1'] + C['lq3'],np.conjugate(ckm),ckm)
    wc['CVRR_eu'] = C['eu']
    wc['CVRR_ed'] = C['ed']
    wc['CVLR_eu'] = C['lu']
    wc['CVLR_ed'] = C['ld']
    wc['CVRL_eu'] = np.einsum('klij->ijkl', C['qe'])
    wc['CVRL_ed'] = np.einsum('mnij,mk,nl->ijkl', C['qe'],np.conjugate(ckm),ckm)
    wc['CSRL_ed'] = np.einsum('ijkm,ml->ijkl',C['ledq'],ckm)
    wc['CSRR_eu'] = -C['lequ1']
    wc['CTRR_eu'] = -C['lequ3']
    wc['CSRL_eu'] = wc['CSRR_ed'] = wc['CTRR_ed'] = np.zeros((3, 3, 3, 3))

    return wc


# translate quark name to LHAPDF flavour index
pdf_flavor = {'d': 1, 'u': 2, 's': 3, 'c': 4, 'b': 5}
fermion_indices = {
    'u': ('u', 0),
    'c': ('u', 1),
    'd': ('d', 0),
    's': ('d', 1),
    'b': ('d', 2),
    'e': ('e', 0),
    'mu': ('e', 1),
    'tau': ('e', 2),
}


def sigma_qqll_partonic(sh, q1, q2, l, wc_eff):
    r"""Total partonic cross section of $q1 q2 \to \ell^+\ell^-$

    Returns $\sigma$ in units of GeV$^{-2}$

    Parameters:
    - `sh`: partonic center of mass energy in GeV$^2$
    - `q1`, `q2`: initial state quarks
    - `l`: final state leptons
    - `wc_eff`: SM + SMEFT amplitude for the process as dictionary with entries corresponding to different Wilson coefficients
    """
    # sh is partonic!
    q, i1 = fermion_indices[q1]
    _q, i2 = fermion_indices[q2]
    if q != _q:
        raise ValueError("Quarks must have the same charge.")
    _, il = fermion_indices[l]
    S = (
        + abs(wc_eff['CVLL_e{}'.format(q)][il, il, i1, i2])**2
        + abs(wc_eff['CVLR_e{}'.format(q)][il, il, i1, i2])**2
        + abs(wc_eff['CVRL_e{}'.format(q)][il, il, i1, i2])**2
        + abs(wc_eff['CVRR_e{}'.format(q)][il, il, i1, i2])**2
        + 3 / 4 * abs(wc_eff['CSRL_e{}'.format(q)][il, il, i1, i2])**2
        + 3 / 4 * abs(wc_eff['CSRR_e{}'.format(q)][il, il, i1, i2])**2
        + 4 * abs(wc_eff['CTRR_e{}'.format(q)][il, il, i1, i2])**2
        # add contributions from h.c. of the effective Lagrangian, (prst) -> (rpts)
        + 3 / 4 * abs(wc_eff['CSRL_e{}'.format(q)][il, il, i2, i1])**2
        + 3 / 4 * abs(wc_eff['CSRR_e{}'.format(q)][il, il, i2, i1])**2
        + 4 * abs(wc_eff['CTRR_e{}'.format(q)][il, il, i2, i1])**2
    )
    return sh / (144 * pi) * S


def dsigma_dtau_qqll(s, tau, l, Q2, wc_eff, par):
    r"""Differential hadronic cross section of $pp\to \ell^+\ell^-$.

    Returns $\frac{d\sigma}{d\tau}$ in units of GeV$^{-2}$.

    Parameters:
    - `s`: hadronic center of mass energy in GeV$^2$
    - `tau`: $\tau$, ratio of partonic to hadronic center of mass energy, dimensionless
    - `l`: lepton flavour, should be 'e', 'mu', or 'tau'
    - `Q2`: factorization scale squared in GeV$^2$
    - `wc_eff`: SM + SMEFT amplitude for the process as dictionary with entries corresponding to different Wilson coefficients
    - `par`: parameter dictionary
    """
    sigma = 0
    members_par = config['PDF set']['dileptons']['members par']
    member = int(par[members_par])
    plumi = partondist.get_parton_lumi(Q2=Q2, member=member)
    for q1 in 'duscb':
        for q2 in 'duscb':
            ud1, i1 = fermion_indices[q1]
            ud2, i2 = fermion_indices[q2]
            if ud1 != ud2:
                continue  # skip cases where q1 and q2 have different charge
            i1 = pdf_flavor[q1]
            i2 = pdf_flavor[q2]
            sh = tau * s
            y = sigma_qqll_partonic(sh, q1, q2, l, wc_eff)
            if y != 0:  # save time if the contribution is anyway zero
                sigma += 2 * plumi.L(i1, -i2, tau) * y
    return sigma


def sigma_qqll_int(s, qmin, qmax, l, Q2, wc_obj, par, scale, newphys=True):
    r"""Integrated hadronic cross section of $pp\to \ell^+\ell^-$.

    Parameters:
    - `s`: hadronic center of mass energy in GeV$^2$
    - `qmin`, `qmax`: bin boundaries in the dileption invariant mass, in GeV
    - `l`: lepton flavour, should be 'e', 'mu', or 'tau'
    - `Q2`: factorization scale squared in GeV$^2$
    - `wc_obj`: Wilson coefficient object
    - `par`: parameter dictionary
    - `scale`: scale at which the Wilson coefficients are evaluated in GeV
    - `newphys`: boolean, whether to include NP contributions or not
    """

    if newphys:
        wc_eff_np = wceff_qqll_np(wc_obj, par, scale)
    else:
        wc_eff_np = {}
    def f(tau):
        wc_eff = wceff_qqll_sm(tau * s, par)
        if newphys:
            # add NP contribution
            wc_eff = add_dict((wc_eff, wc_eff_np))
        return dsigma_dtau_qqll(s, tau, l, Q2, wc_eff, par)
    return nintegrate(f, qmin**2 / s, qmax**2 / s, epsrel=1e-5)


def R_sigma_qqll_int(s, qmin, qmax, l, wc_obj, par):
    r"""Integrated hadronic cross section of $pp\to \ell^+\ell^-$ normalized
    to its SM value.

    Parameters:
    - `s`: hadronic center of mass energy in GeV$^2$
    - `qmin`, `qmax`: bin boundaries in the dileption invariant mass, in GeV
    - `l`: lepton flavour, should be 'e', 'mu', or 'tau'
    - `wc_obj`: Wilson coefficient object
    - `par`: parameter dictionary
    """
    # Renormalization and factorization scale are fixed in the bin
    scale = (qmin + qmax) / 2
    Q2 = scale**2
    sigma_sm = sigma_qqll_int(s, qmin, qmax, l, Q2, wc_obj, par, scale, newphys=False)
    sigma_np = sigma_qqll_int(s, qmin, qmax, l, Q2, wc_obj, par, scale, newphys=True)
    return sigma_np / sigma_sm


def R_sigma_qqll_int_fct_lhc13(l):
    s = 13000**2
    def f(wc_obj, par, qmin, qmax):
        return R_sigma_qqll_int(s, qmin, qmax, l, wc_obj, par)
    return f


# Observable and Prediction instances
_tex = {'e': 'e', 'mu': r'\mu'}

for l in _tex:
    _process_tex = r"pp\to " + _tex[l] + r"^+" + _tex[l] + r"^-"
    _process_taxonomy = r'Process :: contact interactions :: $' + _process_tex + r"$"

    _obs_name = "R_13(pp->" + 2 * l + ")"
    _obs = Observable(_obs_name, arguments=['qmin', 'qmax'])
    _obs.set_description(r"Cross section of $" + _process_tex + r"$ at $\sqrt{s}=13$ TeV normalized to the SM contribution")
    _obs.tex = r"$\text{R}_{13}(" + _process_tex + r")$"
    _obs.add_taxonomy(_process_taxonomy)
    Prediction(_obs_name, R_sigma_qqll_int_fct_lhc13(l))
