from flavio.math.integrate import nintegrate
from flavio.physics.common import add_dict
from flavio.physics import ckm as ckm_flavio
from flavio.classes import Observable, Prediction
from . import partondist
import wilson
from numpy import pi, sqrt
import numpy as np
from flavio.config import config

def F_qqlnu_SM(s, par):
    r"""flavour-independent SM $W-$ contribution to the $\bar u d\to \ell^- \nu_\ell$ amplitude."""
    # parameters
    mW = par['m_W']
    GammaW = 1 / par['tau_W']
    s2w = par['s2w']
    aEW = par['alpha_e']
    e2 = 4 * pi * aEW
    g2=e2/s2w * 0.9608 # correction factor from running g^2 to 1TeV

    return g2/2/(s-mW**2 + 1j*mW*GammaW)


def wceff_qqlnu_sm(s, par):
    ckm = ckm_flavio.get_ckm(par)
    E = np.einsum('ij,kl->ijkl', np.eye(3), np.eye(3))
    wc = {}
    # include CKM matrix here
    wc['CVLL_enuud'] = np.einsum('ijml,km->ijkl',F_qqlnu_SM(s, par)*E,ckm)
    wc['CSRL_enuud'] = wc['CSRR_enuud'] = wc['CTRR_enuud'] = np.zeros((3, 3, 3, 3))

    return wc

def wceff_qqlnu_np(wc_obj, par, scale):
    r"""Returns Wilson coefficients of the effective Lagrangian
    $$\mathcal{L} = C_{e \nu u d}^{\Gamma_1\Gamma_2}
    (\bar e_i\Gamma_1 \nu_j)(\bar u_k\Gamma_2 d_l)$$
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
    wc['CVLL_enuud'] = 2*np.einsum('ijkm,ml->ijkl',C['lq3'],ckm)
    wc['CSRL_enuud'] = np.einsum('jilk',np.conjugate(C['ledq']))
    wc['CSRR_enuud'] = np.einsum('jimk,ml->ijkl',np.conjugate(C['lequ1']),ckm)
    wc['CTRR_enuud'] = np.einsum('jimk,ml->ijkl',np.conjugate(C['lequ3']),ckm)

    return wc

fermion_indices = {'u': 0, 'd': 0, 'c': 1, 's': 1, 't': 2, 'b': 2, 'e': 0, 'mu': 1, 'tau': 2}

def dsigma_dtau_qqlnu_partial(s, tau, umin, umax, q1, q2, l, wc_eff):
    r"""Partial result for the cross section of $\bar q1 q2 \to \ell^-\nu$.

    Returns $\int_{umin}^{umax} \frac{d^2\sigma}{d\tau du}$, integrated analytically over $u=m_T^2/s$.

    Parameters:
    - `s`: Hadronic center of mass energy GeV$^2$
    - `tau`: $\tau$, ratio of partonic to hadronic center of mass energy, dimensionless
    - `umin`, `umax: integration boundaries for the variable $u=m_T^2/s$
    - `q1, q2`: initial state quarks (with q1 up-type)
    - `l`: lepton flavour, should be 'e', 'mu', or 'tau'
    - `wc_eff`: SM + SMEFT amplitude for the process as dictionary with entries corresponding to different Wilson coefficients
    """
    i1 = fermion_indices[q1]
    i2 = fermion_indices[q2]
    il = fermion_indices[l]

    if tau > umax:
        prefac1 = umax * sqrt(1-umax/tau) - umin * sqrt(1-umin/tau)
        prefac2 = tau * ( sqrt(1-umax/tau) - sqrt(1-umin/tau) )
    else:
        prefac1 = - umin * sqrt(1-umin/tau)
        prefac2 = - tau * sqrt(1-umin/tau)

    S = 0
    # sum over neutrino flavours
    for inu in range(3):
        S += s * (
            prefac1 * (
                + 16 * abs(wc_eff['CTRR_enuud'][il, inu, i1, i2])**2
                + abs(wc_eff['CVLL_enuud'][il, inu, i1, i2])**2
            )
            - prefac2 *(
                + 3 * abs(wc_eff['CSRL_enuud'][il, inu, i1, i2])**2
                + 3 * abs(wc_eff['CSRR_enuud'][il, inu, i1, i2])**2
                + 16 * abs(wc_eff['CTRR_enuud'][il, inu, i1, i2])**2
                + 4 * abs(wc_eff['CVLL_enuud'][il, inu, i1, i2])**2
            )
        )

    return S/(576*pi)

pdf_flavor = {'d': 1, 'u': 2, 's': 3, 'c': 4, 'b': 5}

def dsigma_dtau_qqlnu(s, tau, umin, umax, l, wc_eff, par, Q2):
    r"""Differential hadronic cross section of $pp\to \ell\nu$.

    Returns ${d\sigma}{d\tau}$ in units of GeV$^{-2}$.

    Parameters:
    - `s`: hadronic center of mass energy in GeV$^2$
    - `tau`: $\tau$, ratio of partonic to hadronic center of mass energy, dimensionless
    - `umin`, `umax`: integration boundaries for the variable $u=m_T^2/s$
    - `l`: lepton flavour, should be 'e', 'mu', or 'tau'
    - `wc_eff`: SM + SMEFT amplitude for the process as dictionary with entries corresponding to different Wilson coefficients
    - `par`: parameter dictionary
    - `Q2`: factorization scale squared in GeV$^2$
    """
    sigma = 0
    members_par = config['PDF set']['dileptons']['members par']
    member = int(par[members_par])
    plumi = partondist.get_parton_lumi(Q2=Q2, member=member)
    for q1 in 'uc':
        for q2 in 'dsb':
            i1 = pdf_flavor[q1]
            i2 = pdf_flavor[q2]
            y = dsigma_dtau_qqlnu_partial(s, tau, umin, umax, q1, q2, l, wc_eff)
            if y != 0:  # save time if the contribution is anyway zero
                sigma += 2 * plumi.L(i1, -i2, tau) * y # $W^+$ contribution
                sigma += 2 * plumi.L(i2, -i1, tau) * y # $W^-$ contribution
    return sigma

def sigma_qqlnu_int(s, mTmin, mTmax, l, wc_obj, par, Q2, scale, newphys=True):
    r"""Integrated hadronic cross section of $pp\to \ell\nu$.

    Parameters:
    - `s`: hadronic center of mass energy in GeV$^2$
    - `mTmin`, `mTmax`: bin boundaries in transverse mass mT in GeV
    - `l`: lepton flavour, should be 'e', 'mu', or 'tau'
    - `wc_obj`: Wilson coefficient object
    - `par`: parameter dictionary
    - `Q2`: factorization scale squared in GeV$^2$
    - `scale`: scale at which the Wilson coefficients are evaluated in GeV
    - `newphys`: boolean, whether to include NP contributions or not
    """
    # change of variables to $u=m_T^2/s$
    umin = mTmin**2/s
    umax = mTmax**2/s

    if newphys:
        wc_eff_np = wceff_qqlnu_np(wc_obj, par, scale)
    else:
        wc_eff_np = {}
    def f(tau):
        wc_eff = wceff_qqlnu_sm(tau * s, par)
        if newphys:
            # add NP contribution
            wc_eff = add_dict((wc_eff, wc_eff_np))
        return dsigma_dtau_qqlnu(s, tau, umin, umax, l, wc_eff, par,  Q2)
    # integrate over tau from umin to 1
    return nintegrate(f, umin, 1, epsrel=1e-5)

def R_sigma_qqlnu_int(s, mTmin, mTmax, l, wc_obj, par):
    r"""Integrated hadronic cross section of $pp\to \ell \nu$ normalized
    to its SM value.

    Parameters:
    - `s`: hadronic center of mass energy in GeV$^2$
    - `mTmin`, `mTmax`: bin boundaries in transverse mass mT in GeV
    - `l`: lepton flavour, should be 'e', 'mu', or 'tau'
    - `wc_obj`: Wilson coefficient object
    - `par`: parameter dictionary
    """
    # Renormalization and factorization scale are fixed in the bin
    scale = (mTmin + mTmax)/2
    Q2 = scale**2
    sigma_sm = sigma_qqlnu_int(s, mTmin, mTmax, l, wc_obj, par, Q2, scale, newphys=False)
    sigma_np = sigma_qqlnu_int(s, mTmin, mTmax, l, wc_obj, par, Q2, scale, newphys=True)
    return sigma_np / sigma_sm


def R_sigma_qqlnu_int_fct_lhc13(l):
    s = 13000**2
    def f(wc_obj, par, mTmin, mTmax):
        return R_sigma_qqlnu_int(s, mTmin, mTmax, l, wc_obj, par)
    return f


# Observable and Prediction instances
_tex = {'e': 'e', 'mu': r'\mu'}

for l in _tex:
    _process_tex = r"pp\to " + _tex[l]+r"\nu"
    _process_taxonomy = r'Process :: contact interactions :: $' + _process_tex + r"$"

    _obs_name = "R_13(pp->" + l + "nu)"
    _obs = Observable(_obs_name, arguments=['mTmin', 'mTmax'])
    _obs.set_description(r"Cross section of $" + _process_tex + r"$ at $\sqrt{s}=13$ TeV normalized to the SM contribution")
    _obs.tex = r"$\text{R}_{13}(" + _process_tex + r")$"
    _obs.add_taxonomy(_process_taxonomy)
    Prediction(_obs_name, R_sigma_qqlnu_int_fct_lhc13(l))
