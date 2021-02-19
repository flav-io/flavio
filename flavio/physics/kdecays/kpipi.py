r"""Functions for observables in $K\to\pi\pi$ decays, in particular
$\epsilon'/\epsilon$."""

import flavio
from flavio.classes import Prediction, Observable
from flavio.config import config
from flavio.physics import ckm
from flavio.physics.kdecays.wilsoncoefficients import wilsoncoefficients_sm_fourquark
from math import sqrt



def Kpipi_matrixelements_SM(par, scale):
    r"""Return the $K\to\pi\pi$ matrix elements of the SM operators in the
    traditional 10-operator basis.

    Returns a dictionary with keys 0 and 2 corresponding to the $\Delta I=1/2$
    and 3/2 matrix elements, respectively, and values being again dictionaries
    with the keys being the operator index and the value being the matrix
    element in units of GeV³, using the same normalization as in
    arXiv:1502.00263."""
    M0 = {}
    M2 = {}
    for i in [3, 4, 5, 6, 7, 8, 9]:
        M0[str(i)] = par['Kpipi M0 {}'.format(i)]
    for i in [7, 8, 9]:
        M2[str(i)] = par['Kpipi M2 {}'.format(i)]
    # Exact relations
    for i in [3, 4, 5, 6]:
        M2[str(i)] = 0
    M0['1'] = 1 / 3 * (M0['3'] + 2 * M0['9'])
    M0['2'] = 1 / 3 * (-2 * M0['3'] + 3 * M0['4'] + 2 * M0['9'])
    M0['10'] = -M0['3'] + M0['4'] + M0['9']
    M2['1'] = 2 / 3 * M2['9']
    M2['2'] = M2['1']
    M2['10'] = M2['9']
    return {0: M0, 2: M2}


def Kpipi_matrixelements_NP(par, scale):
    r"""Return the $K\to\pi\pi$ matrix elements of all $s\to d$ operators in the
    flavio basis.

    Returns a dictionary with keys 0 and 2 corresponding to the $\Delta I=1/2$
    and 3/2 matrix elements, respectively, and values being again dictionaries
    with the keys being the operator name and the value being the matrix
    element in units of GeV³, using the same normalization as in
    arXiv:1502.00263."""
    MSM = Kpipi_matrixelements_SM(par, scale)
    M = {0: {}, 2: {}}
    ms = flavio.physics.running.running.get_ms(par, scale, nf_out=3)
    # follows appendix A4 of Aebischer/Buras/Gerard arXiv:1807.01709
    for i in (0, 2):
        M[i]['CVLL_sduu']  =  MSM[i]['1'] / 4
        M[i]['CVLLt_sduu'] =  MSM[i]['2'] / 4
        M[i]['CVLR_sduu'] = (MSM[i]['5'] / 3 + 2 * MSM[i]['7'] / 3) / 4
        M[i]['CVLRt_sduu'] = (MSM[i]['6'] / 3 + 2 * MSM[i]['8'] / 3) / 4
        M[i]['CVLL_sddd'] = (2 * MSM[i]['3'] / 3 - 2 * MSM[i]['9'] / 3) / 4
        M[i]['CVLR_sddd'] = (2 * MSM[i]['5'] / 3 - 2 * MSM[i]['7'] / 3) / 4
        M[i]['CSRL_sddd'] = (MSM[i]['6'] / 3 - MSM[i]['8'] / 3) / 4
        M[i]['CSRR_sddd']  =  par['Kpipi M{} SLL2_d'.format(i)]
        M[i]['CTRR_sddd']  = -8 * par['Kpipi M{} SLL1_d'.format(i)] -4 * par['Kpipi M{} SLL2_d'.format(i)]
    for i in (2, ):
        # isospin relations valid for I=2 amplitude
        M[i]['CSRL_sduu'] = -M[i]['CSRL_sddd']
        M[i]['CSRR_sduu'] = -M[i]['CSRR_sddd']
        M[i]['CTRR_sduu'] = -M[i]['CTRR_sddd']
        M[i]['CSRLt_sduu'] = 1 / 2 * M[i]['CVLR_sddd']  # -1 from isospin, -1/2 from Fierz
        M[i]['CSRRt_sduu'] = -(-1 / 2 * M[i]['CSRR_sddd'] - 1 / 8 * M[i]['CTRR_sddd'])
        M[i]['CTRRt_sduu'] = -(-6 * M[i]['CSRR_sddd'] + 1 / 2 * M[i]['CTRR_sddd'])
        M[i]['C8_sd']      = 0
    for i in (0, ):
        M[i]['CSRL_sduu']  =  par['Kpipi M{} SLR2_u'.format(i)]
        M[i]['CSRLt_sduu'] =  par['Kpipi M{} SLR1_u'.format(i)]
        M[i]['CSRR_sduu']  =  par['Kpipi M{} SLL2_u'.format(i)]
        M[i]['CSRRt_sduu'] =  par['Kpipi M{} SLL1_u'.format(i)]
        M[i]['CTRR_sduu']  = -par['Kpipi M{} SLL4_u'.format(i)]
        M[i]['CTRRt_sduu'] = -par['Kpipi M{} SLL3_u'.format(i)]
        M[i]['C8_sd']      = -ms * par['Kpipi M0 g-'] / 2
    for i in (0, 2):
        M[i]['CVRR_sduu']  = -M[i]['CVLL_sduu']
        M[i]['CVRRt_sduu'] = -M[i]['CVLLt_sduu']
        M[i]['CVRR_sddd']  = -M[i]['CVLL_sddd']
        M[i]['CVRL_sduu']  = -M[i]['CVLR_sduu']
        M[i]['CVRLt_sduu'] = -M[i]['CVLRt_sduu']
        M[i]['CVRL_sddd']  = -M[i]['CVLR_sddd']
        M[i]['CSLR_sduu']  = -M[i]['CSRL_sduu']
        M[i]['CSLRt_sduu'] = -M[i]['CSRLt_sduu']
        M[i]['CSLR_sddd']  = -M[i]['CSRL_sddd']
        M[i]['CSLL_sduu']  = -M[i]['CSRR_sduu']
        M[i]['CSLLt_sduu'] = -M[i]['CSRRt_sduu']
        M[i]['CSLL_sddd']  = -M[i]['CSRR_sddd']
        M[i]['CTLL_sduu']  = -M[i]['CTRR_sduu']
        M[i]['CTLLt_sduu'] = -M[i]['CTRRt_sduu']
        M[i]['CTLL_sddd']  = -M[i]['CTRR_sddd']
        M[i]['C8p_sd']     = -M[i]['C8_sd']
    return M


def Kpipi_amplitudes_SM(par,
                       include_VmA=True, include_VpA=True,
                       scale_ImA0EW=False):
    r"""Compute the SM contribution to the two isospin amplitudes of
    the $K\to\pi\pi$ transition."""
    scale = config['renormalization scale']['kpipi']
    pref = par['GF'] / sqrt(2) * ckm.xi('u', 'ds')(par)  # GF/sqrt(2) Vus* Vud
    me = Kpipi_matrixelements_SM(par, scale)
    # Wilson coefficients
    wc = wilsoncoefficients_sm_fourquark(par, scale)
    tau = -ckm.xi('t', 'ds')(par) / ckm.xi('u', 'ds')(par)
    k = [1, 2]
    if include_VmA:
        k = k + [3, 4, 9, 10]
    if include_VpA:
        k = k + [5, 6, 7, 8]
    A = {0: 0, 2: 0}
    for i in [0, 2]:
        for j in k:
            m = me[i][str(j)]
            yj = wc.get('y{}'.format(j), 0)
            zj = wc.get('z{}'.format(j), 0)
            dA = pref * m * (zj + tau * yj)
            if scale_ImA0EW and i == 0 and j in [7, 8, 9, 10]:
                b = 1 / par['epsp a'] / (1 - par['Omegahat_eff'])
                dA = dA.real + 1j * b * dA.imag
            A[i] += dA
    return A


def Kpipi_amplitudes_NP(wc_obj, par):
    r"""Compute the new physics contribution to the two isospin amplitudes
    of the $K\to\pi\pi$ transition."""
    scale = config['renormalization scale']['kpipi']
    pref = 4 * par['GF'] / sqrt(2) * ckm.xi('t', 'ds')(par)  # 4GF/sqrt(2) Vts* Vtd
    me = Kpipi_matrixelements_NP(par, scale)
    wc = wc_obj.get_wc(sector='sd', scale=scale, par=par, eft='WET-3')
    A = {0: 0, 2: 0}
    for i in [0, 2]:
        for j, m in me[i].items():
            A[i] += -pref * m * complex(wc[j]).conjugate()  # conjugate!
    return A

def epsprime_SM(par):
    r"""Compute the SM contribution to $\epsilon'/\epsilon$, including
    isospin breaking corrections."""
    a = par['epsp a']
    A = Kpipi_amplitudes_SM(par)
    ImA0 = A[0].imag
    ImA2 = A[2].imag
    ReA0 = par['ReA0(K->pipi)']
    ReA2 = par['ReA2(K->pipi)']
     # eq. (19) of arXiv:1507.06345
    flavio.citations.register("Buras:2015yba")
    return (-par['omega+'] / (sqrt(2) * par['eps_K'])
            * (ImA0 / ReA0 * (1 - par['Omegahat_eff'])
               - 1 / a * ImA2 / ReA2).real)


def epsprime_NP(wc_obj, par):
    r"""Compute the NP contribution to $\epsilon'/\epsilon$."""
    # Neglecting isospin breaking corrections!
    A = Kpipi_amplitudes_NP(wc_obj, par)
    ImA0 = A[0].imag
    ImA2 = A[2].imag
    ReA0 = par['ReA0(K->pipi)']
    ReA2 = par['ReA2(K->pipi)']
    a = par['epsp a']  # eq. (16)
    # dividing by a to remove the isospin brk corr in omega+, cf. (16) in 1507.06345
    flavio.citations.register("Buras:2015yba")
    return (-par['omega+'] / a / (sqrt(2) * par['eps_K'])
            * (ImA0 / ReA0 - ImA2 / ReA2).real)

def epsprime(wc_obj, par):
    r"""Compute $\epsilon'/\epsilon$, parametrizing direct CPV in
    $K\to\pi\pi$."""
    return epsprime_SM(par) + epsprime_NP(wc_obj, par)


# Observable and Prediction instances
o = Observable('epsp/eps')
o.tex = r"$\epsilon^\prime/\epsilon$"
Prediction('epsp/eps', epsprime)
o.set_description(r"Direct CP violation parameter")
o.add_taxonomy(r'Process :: $s$ hadron decays :: Non-leptonic decays :: $K\to \pi\pi$')
