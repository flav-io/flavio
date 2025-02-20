r"""$B^+ / B_d$ lifetime ratio"""

import flavio
from flavio.physics import ckm
from flavio.physics.running.running import get_alpha_s
from flavio.config import config
from .wilsoncoefficients import wcsm_nf5
import numpy as np
from math import pi


def run_lifetime_bag_parameters(par, scale):
    if scale < config['RGE thresholds']['mc'] or scale > 4.5:
        raise ValueError("Scale for running the B lifetime bag parameters must be between mc and 4.5 GeV.")
    alpha_s = get_alpha_s(par, scale)
    alpha_s_0 = get_alpha_s(par, 1.5)
    eta = alpha_s / alpha_s_0

    beta_0_nf4 = 11 - (2/3)*4

    B1qtilde_0 = par["bag_lifetime_B1qtilde"]
    B2qtilde_0 = par["bag_lifetime_B2qtilde"]
    B3qtilde_0 = par["bag_lifetime_B3qtilde"]
    B4qtilde_0 = par["bag_lifetime_B4qtilde"]
    B5qtilde_0 = par["bag_lifetime_B5qtilde"]
    B6qtilde_0 = par["bag_lifetime_B6qtilde"]
    B7qtilde_0 = par["bag_lifetime_B7qtilde"]
    B8qtilde_0 = par["bag_lifetime_B8qtilde"]
    deltaqq1tilde_0 = par["bag_lifetime_delta^ud_1tilde"]
    deltaqq2tilde_0 = par["bag_lifetime_delta^ud_2tilde"]
    deltaqq3tilde_0 = par["bag_lifetime_delta^ud_3tilde"]
    deltaqq4tilde_0 = par["bag_lifetime_delta^ud_4tilde"]

    # We only evolve the SM bag parameters B1-4, matching what is done in 2208.02643
    gamma_0_D = np.array((8,8,-1,-1))
    V = np.array((
        (0,3/4,0,-6),
        (3/4,0,-6,0),
        (0,1,0,1),
        (1,0,1,0)
    ))
    invV = np.array((
        (0,4/27,0,8/9),
        (4/27,0,8/9,0),
        (0,-4/27,0,1/9),
        (-4/27,0,1/9,0)
    ))
    # Evolution matrix from 1.5 GeV to scale
    U = V @ np.diag(eta**(gamma_0_D / (2*beta_0_nf4))) @ invV

    B_0 = np.array((B1qtilde_0, B2qtilde_0, B3qtilde_0, B4qtilde_0))
    B1qtilde, B2qtilde, B3qtilde, B4qtilde = U @ B_0

    return {
        "bag_lifetime_B1qtilde": B1qtilde,
        "bag_lifetime_B2qtilde": B2qtilde,
        "bag_lifetime_B3qtilde": B3qtilde,
        "bag_lifetime_B4qtilde": B4qtilde,
        "bag_lifetime_B5qtilde": B5qtilde_0,
        "bag_lifetime_B6qtilde": B6qtilde_0,
        "bag_lifetime_B7qtilde": B7qtilde_0,
        "bag_lifetime_B8qtilde": B8qtilde_0,
        "bag_lifetime_deltaqq1tilde": deltaqq1tilde_0,
        "bag_lifetime_deltaqq2tilde": deltaqq2tilde_0,
        "bag_lifetime_deltaqq3tilde": deltaqq3tilde_0,
        "bag_lifetime_deltaqq4tilde": deltaqq4tilde_0,
    }


def tau_Bp_over_tau_Bd_SM(par):
    r"""Sm contribution to the ratio of the B+ to Bd lifetimes."""
    scale = config['renormalization scale']['b lifetime ratios']
    if scale != 4.5:
        raise ValueError("The SM prediction for the B+ / Bd lifetime ratio is only available at the scale 4.5 GeV.")
    bag_params_dict = run_lifetime_bag_parameters(par, scale)
    B1qtilde = bag_params_dict["bag_lifetime_B1qtilde"]
    B2qtilde = bag_params_dict["bag_lifetime_B2qtilde"]
    B3qtilde = bag_params_dict["bag_lifetime_B3qtilde"]
    B4qtilde = bag_params_dict["bag_lifetime_B4qtilde"]
    deltaqq1tilde = bag_params_dict["bag_lifetime_deltaqq1tilde"]
    deltaqq2tilde = bag_params_dict["bag_lifetime_deltaqq2tilde"]
    deltaqq3tilde = bag_params_dict["bag_lifetime_deltaqq3tilde"]
    deltaqq4tilde = bag_params_dict["bag_lifetime_deltaqq4tilde"]

    # Phenomenological formula from Lenz:2022rbq
    flavio.citations.register("Lenz:2022rbq")
    ratio_SM = 1 + 0.059 * B1qtilde + 0.005 * B2qtilde - 0.674 * B3qtilde + 0.160 * B4qtilde \
                 - 0.025 * deltaqq1tilde + 0.002 * deltaqq2tilde + 0.591 * deltaqq3tilde - 0.152 * deltaqq4tilde \
                 - 0.007
    return ratio_SM

def gamma_BSM_dim6(wc_obj, par, meson):
    WE = weak_exchange(wc_obj, par, meson)
    PI = pauli_interference(wc_obj, par, meson)
    return WE + PI


def siegen_basis_wcs(wc_obj, par, sector):
    scale = config['renormalization scale']['b lifetime ratios']
    wc_sm = wcsm_nf5(scale)
    wcxf_sector = "db" if sector == "dbcc" else sector
    flavio_wc_bsm = wc_obj.get_wc(sector=wcxf_sector, scale=scale, par=par)

    CSM = np.zeros(20)
    CSM[0] = -1/6 * wc_sm[0] + wc_sm[1]
    CSM[1] = 1/2 * wc_sm[0]

    CNP = np.zeros(20, dtype=complex)
    siegen_wc_order = (
        "VLL", "VLLt", "VRL", "VRLt", "SLR", "SLRt", "SRR", "SRRt", "TRR", "TRRt",
        "VRR", "VRRt", "VLR", "VLRt", "SRL", "SRLt", "SLL", "SLLt", "TLL", "TLLt"
    )
    flavio_sector_name_mapping = {"sbcu": "bcus", "dbcu": "bcud", "dbcc": "bdcc"}
    flavio_sector_name = flavio_sector_name_mapping[sector]
    for i, name in enumerate(siegen_wc_order):
        CNP[i] = flavio_wc_bsm[f"C{name}_{flavio_sector_name}"]
    return (CSM, CNP)


def lifetimematrixelements(par, meson, scale):
    r"""Returns a dictionary with the values of the matrix elements of the
    $\Delta B=0$ operators.

    Note that the normalisation factor 1/2M is included here,
    so the returned values of the matrix elements are really
    $$\langle O \rangle = \frac{\langle B_q | O | B_q\rangle}{2M_B}$$
    """
    mM = par['m_'+meson]
    fM = par['f_'+meson]
    bag_params_dict = run_lifetime_bag_parameters(par, scale)
    BM = lambda i: bag_params_dict[f"bag_lifetime_B{i}qtilde"]
    me = {}
    for i in range(1,9):
        me[f"{i}"]  = fM**2 * mM * BM(i) / 2
        me[f"{i}p"] = fM**2 * mM * BM(i) / 2
    return me


def weak_exchange(wc_obj, par, meson):
    r"""BSM weak exchange contributions. For now, only from b->c ubar d
    (dbcu sector), b->c ubar s (sbcu sector), and b->c cbar d (db sector) operators."""
    # For B+, no WE contribution from our considered sectors
    if meson == "B+":
        return 0
    # So now Bd case only: dbcu and dbcc operators contribute

    GF = par["GF"]
    mb = flavio.physics.running.running.get_mb_KS(par, 1)
    mc = par["m_c"]
    rho = mc**2 / mb**2
    V = ckm.get_ckm(par)

    me = lifetimematrixelements(par, meson, config["renormalization scale"]["b lifetime ratios"])
    flavio.citations.register("Lenz:2022pgw")
    # See eqs 2.28 - 2.33 in Lenz:2022pgw
    A_WE_cu = np.array((
        ((-((2 + rho)*me["1"]) + 2*(me["2"] + 2*rho*me["2"] - 3*(2 + rho)*me["3"] + 6*(me["4"] + 2*rho*me["4"])))/6,-((2 + rho)*me["1"])/2 + me["2"] + 2*rho*me["2"],-(rho**0.5*(me["2"] + 6*me["4"])),-3*rho**0.5*me["2"],0,0,0,0,0,0,0,0,0,0,((2 + rho)*me["5p"] - 2*(me["6p"] + 2*rho*me["6p"] - 3*(2 + rho)*me["7p"] + 6*(me["8p"] + 2*rho*me["8p"])))/12,((2 + rho)*me["5p"] - 2*(me["6p"] + 2*rho*me["6p"]))/4,(rho**0.5*(me["5p"] - 2*(me["6p"] - 3*me["7p"] + 6*me["8p"])))/4,(3*rho**0.5*(me["5p"] - 2*me["6p"]))/4,-(rho**0.5*(me["5p"] + 2*(me["6p"] + 3*me["7p"] + 6*me["8p"]))),-3*rho**0.5*(me["5p"] + 2*me["6p"])),
        (-((2 + rho)*me["1"])/2 + me["2"] + 2*rho*me["2"],(-3*(2 + rho)*me["1"])/2 + 3*(1 + 2*rho)*me["2"],-3*rho**0.5*me["2"],-9*rho**0.5*me["2"],0,0,0,0,0,0,0,0,0,0,((2 + rho)*me["5p"] - 2*(me["6p"] + 2*rho*me["6p"]))/4,(3*((2 + rho)*me["5p"] - 2*(me["6p"] + 2*rho*me["6p"])))/4,(3*rho**0.5*(me["5p"] - 2*me["6p"]))/4,(9*rho**0.5*(me["5p"] - 2*me["6p"]))/4,-3*rho**0.5*(me["5p"] + 2*me["6p"]),-9*rho**0.5*(me["5p"] + 2*me["6p"])),
        (-(rho**0.5*(me["2"] + 6*me["4"])),-3*rho**0.5*me["2"],2*(me["2"] + 6*me["4"]),6*me["2"],0,0,0,0,0,0,0,0,0,0,(rho**0.5*(me["6p"] + 6*me["8p"]))/2,(3*rho**0.5*me["6p"])/2,(me["6p"] + 6*me["8p"])/2,(3*me["6p"])/2,6*(me["6p"] + 6*me["8p"]),18*me["6p"]),
        (-3*rho**0.5*me["2"],-9*rho**0.5*me["2"],6*me["2"],18*me["2"],0,0,0,0,0,0,0,0,0,0,(3*rho**0.5*me["6p"])/2,(9*rho**0.5*me["6p"])/2,(3*me["6p"])/2,(9*me["6p"])/2,18*me["6p"],54*me["6p"]),
        (0,0,0,0,(-((2 + rho)*me["1"]) + 2*(me["2"] + 2*rho*me["2"] - 3*(2 + rho)*me["3"] + 6*(me["4"] + 2*rho*me["4"])))/24,(-((2 + rho)*me["1"]) + 2*(1 + 2*rho)*me["2"])/8,-(rho**0.5*(me["1"] - 2*(me["2"] - 3*me["3"] + 6*me["4"])))/8,(-3*rho**0.5*(me["1"] - 2*me["2"]))/8,(rho**0.5*(me["1"] + 2*(me["2"] + 3*me["3"] + 6*me["4"])))/2,(3*rho**0.5*(me["1"] + 2*me["2"]))/2,((2 + rho)*me["5p"] - 2*(me["6p"] + 2*rho*me["6p"] - 3*(2 + rho)*me["7p"] + 6*(me["8p"] + 2*rho*me["8p"])))/12,((2 + rho)*me["5p"] - 2*(me["6p"] + 2*rho*me["6p"]))/4,(rho**0.5*(me["6p"] + 6*me["8p"]))/2,(3*rho**0.5*me["6p"])/2,0,0,0,0,0,0),
        (0,0,0,0,(-((2 + rho)*me["1"]) + 2*(1 + 2*rho)*me["2"])/8,(-3*((2 + rho)*me["1"] - 2*(me["2"] + 2*rho*me["2"])))/8,(-3*rho**0.5*(me["1"] - 2*me["2"]))/8,(-9*rho**0.5*(me["1"] - 2*me["2"]))/8,(3*rho**0.5*(me["1"] + 2*me["2"]))/2,(9*rho**0.5*(me["1"] + 2*me["2"]))/2,((2 + rho)*me["5p"] - 2*(me["6p"] + 2*rho*me["6p"]))/4,(3*((2 + rho)*me["5p"] - 2*(me["6p"] + 2*rho*me["6p"])))/4,(3*rho**0.5*me["6p"])/2,(9*rho**0.5*me["6p"])/2,0,0,0,0,0,0),
        (0,0,0,0,-(rho**0.5*(me["1"] - 2*(me["2"] - 3*me["3"] + 6*me["4"])))/8,(-3*rho**0.5*(me["1"] - 2*me["2"]))/8,(-((1 + 2*rho)*me["1"]) + 2*(2 + rho)*me["2"] - 6*(me["3"] + 2*rho*me["3"] - 2*(2 + rho)*me["4"]))/24,(-((1 + 2*rho)*me["1"]) + 2*(2 + rho)*me["2"])/8,(me["1"] + 2*rho*me["1"] - 2*(-4 + rho)*me["2"] + 6*(me["3"] + 2*rho*me["3"] + 8*me["4"] - 2*rho*me["4"]))/6,(0.5 + rho)*me["1"] - (-4 + rho)*me["2"],(rho**0.5*(me["5p"] - 2*(me["6p"] - 3*me["7p"] + 6*me["8p"])))/4,(3*rho**0.5*(me["5p"] - 2*me["6p"]))/4,(me["6p"] + 6*me["8p"])/2,(3*me["6p"])/2,0,0,0,0,0,0),
        (0,0,0,0,(-3*rho**0.5*(me["1"] - 2*me["2"]))/8,(-9*rho**0.5*(me["1"] - 2*me["2"]))/8,(-((1 + 2*rho)*me["1"]) + 2*(2 + rho)*me["2"])/8,(-3*(me["1"] + 2*rho*me["1"] - 2*(2 + rho)*me["2"]))/8,(0.5 + rho)*me["1"] - (-4 + rho)*me["2"],(3*(me["1"] + 2*rho*me["1"] - 2*(-4 + rho)*me["2"]))/2,(3*rho**0.5*(me["5p"] - 2*me["6p"]))/4,(9*rho**0.5*(me["5p"] - 2*me["6p"]))/4,(3*me["6p"])/2,(9*me["6p"])/2,0,0,0,0,0,0),
        (0,0,0,0,(rho**0.5*(me["1"] + 2*(me["2"] + 3*me["3"] + 6*me["4"])))/2,(3*rho**0.5*(me["1"] + 2*me["2"]))/2,(me["1"] + 2*rho*me["1"] - 2*(-4 + rho)*me["2"] + 6*(me["3"] + 2*rho*me["3"] + 8*me["4"] - 2*rho*me["4"]))/6,(0.5 + rho)*me["1"] - (-4 + rho)*me["2"],(-2*(me["1"] + 2*rho*me["1"] - 2*(14 + rho)*me["2"] + 6*(1 + 2*rho)*me["3"] - 12*(14 + rho)*me["4"]))/3,-2*(1 + 2*rho)*me["1"] + 4*(14 + rho)*me["2"],-(rho**0.5*(me["5p"] + 2*(me["6p"] + 3*me["7p"] + 6*me["8p"]))),-3*rho**0.5*(me["5p"] + 2*me["6p"]),6*(me["6p"] + 6*me["8p"]),18*me["6p"],0,0,0,0,0,0),
        (0,0,0,0,(3*rho**0.5*(me["1"] + 2*me["2"]))/2,(9*rho**0.5*(me["1"] + 2*me["2"]))/2,(0.5 + rho)*me["1"] - (-4 + rho)*me["2"],(3*(me["1"] + 2*rho*me["1"] - 2*(-4 + rho)*me["2"]))/2,-2*(1 + 2*rho)*me["1"] + 4*(14 + rho)*me["2"],-6*(me["1"] + 2*rho*me["1"] - 2*(14 + rho)*me["2"]),-3*rho**0.5*(me["5p"] + 2*me["6p"]),-9*rho**0.5*(me["5p"] + 2*me["6p"]),18*me["6p"],54*me["6p"],0,0,0,0,0,0),
        (0,0,0,0,((2 + rho)*me["5"] - 2*(me["6"] + 2*rho*me["6"] - 3*(2 + rho)*me["7"] + 6*(me["8"] + 2*rho*me["8"])))/12,((2 + rho)*me["5"] - 2*(me["6"] + 2*rho*me["6"]))/4,(rho**0.5*(me["5"] - 2*(me["6"] - 3*me["7"] + 6*me["8"])))/4,(3*rho**0.5*(me["5"] - 2*me["6"]))/4,-(rho**0.5*(me["5"] + 2*(me["6"] + 3*me["7"] + 6*me["8"]))),-3*rho**0.5*(me["5"] + 2*me["6"]),(-((2 + rho)*me["1p"]) + 2*(me["2p"] + 2*rho*me["2p"] - 3*(2 + rho)*me["3p"] + 6*(me["4p"] + 2*rho*me["4p"])))/6,-((2 + rho)*me["1p"])/2 + me["2p"] + 2*rho*me["2p"],-(rho**0.5*(me["2p"] + 6*me["4p"])),-3*rho**0.5*me["2p"],0,0,0,0,0,0),
        (0,0,0,0,((2 + rho)*me["5"] - 2*(me["6"] + 2*rho*me["6"]))/4,(3*((2 + rho)*me["5"] - 2*(me["6"] + 2*rho*me["6"])))/4,(3*rho**0.5*(me["5"] - 2*me["6"]))/4,(9*rho**0.5*(me["5"] - 2*me["6"]))/4,-3*rho**0.5*(me["5"] + 2*me["6"]),-9*rho**0.5*(me["5"] + 2*me["6"]),-((2 + rho)*me["1p"])/2 + me["2p"] + 2*rho*me["2p"],(-3*(2 + rho)*me["1p"])/2 + 3*(1 + 2*rho)*me["2p"],-3*rho**0.5*me["2p"],-9*rho**0.5*me["2p"],0,0,0,0,0,0),
        (0,0,0,0,(rho**0.5*(me["6"] + 6*me["8"]))/2,(3*rho**0.5*me["6"])/2,(me["6"] + 6*me["8"])/2,(3*me["6"])/2,6*(me["6"] + 6*me["8"]),18*me["6"],-(rho**0.5*(me["2p"] + 6*me["4p"])),-3*rho**0.5*me["2p"],2*(me["2p"] + 6*me["4p"]),6*me["2p"],0,0,0,0,0,0),
        (0,0,0,0,(3*rho**0.5*me["6"])/2,(9*rho**0.5*me["6"])/2,(3*me["6"])/2,(9*me["6"])/2,18*me["6"],54*me["6"],-3*rho**0.5*me["2p"],-9*rho**0.5*me["2p"],6*me["2p"],18*me["2p"],0,0,0,0,0,0),
        (((2 + rho)*me["5"] - 2*(me["6"] + 2*rho*me["6"] - 3*(2 + rho)*me["7"] + 6*(me["8"] + 2*rho*me["8"])))/12,((2 + rho)*me["5"] - 2*(me["6"] + 2*rho*me["6"]))/4,(rho**0.5*(me["6"] + 6*me["8"]))/2,(3*rho**0.5*me["6"])/2,0,0,0,0,0,0,0,0,0,0,(-((2 + rho)*me["1p"]) + 2*(me["2p"] + 2*rho*me["2p"] - 3*(2 + rho)*me["3p"] + 6*(me["4p"] + 2*rho*me["4p"])))/24,(-((2 + rho)*me["1p"]) + 2*(1 + 2*rho)*me["2p"])/8,-(rho**0.5*(me["1p"] - 2*(me["2p"] - 3*me["3p"] + 6*me["4p"])))/8,(-3*rho**0.5*(me["1p"] - 2*me["2p"]))/8,(rho**0.5*(me["1p"] + 2*(me["2p"] + 3*me["3p"] + 6*me["4p"])))/2,(3*rho**0.5*(me["1p"] + 2*me["2p"]))/2),
        (((2 + rho)*me["5"] - 2*(me["6"] + 2*rho*me["6"]))/4,(3*((2 + rho)*me["5"] - 2*(me["6"] + 2*rho*me["6"])))/4,(3*rho**0.5*me["6"])/2,(9*rho**0.5*me["6"])/2,0,0,0,0,0,0,0,0,0,0,(-((2 + rho)*me["1p"]) + 2*(1 + 2*rho)*me["2p"])/8,(-3*((2 + rho)*me["1p"] - 2*(me["2p"] + 2*rho*me["2p"])))/8,(-3*rho**0.5*(me["1p"] - 2*me["2p"]))/8,(-9*rho**0.5*(me["1p"] - 2*me["2p"]))/8,(3*rho**0.5*(me["1p"] + 2*me["2p"]))/2,(9*rho**0.5*(me["1p"] + 2*me["2p"]))/2),
        ((rho**0.5*(me["5"] - 2*(me["6"] - 3*me["7"] + 6*me["8"])))/4,(3*rho**0.5*(me["5"] - 2*me["6"]))/4,(me["6"] + 6*me["8"])/2,(3*me["6"])/2,0,0,0,0,0,0,0,0,0,0,-(rho**0.5*(me["1p"] - 2*(me["2p"] - 3*me["3p"] + 6*me["4p"])))/8,(-3*rho**0.5*(me["1p"] - 2*me["2p"]))/8,(-((1 + 2*rho)*me["1p"]) + 2*(2 + rho)*me["2p"] - 6*(me["3p"] + 2*rho*me["3p"] - 2*(2 + rho)*me["4p"]))/24,(-((1 + 2*rho)*me["1p"]) + 2*(2 + rho)*me["2p"])/8,(me["1p"] + 2*rho*me["1p"] - 2*(-4 + rho)*me["2p"] + 6*(me["3p"] + 2*rho*me["3p"] + 8*me["4p"] - 2*rho*me["4p"]))/6,(0.5 + rho)*me["1p"] - (-4 + rho)*me["2p"]),
        ((3*rho**0.5*(me["5"] - 2*me["6"]))/4,(9*rho**0.5*(me["5"] - 2*me["6"]))/4,(3*me["6"])/2,(9*me["6"])/2,0,0,0,0,0,0,0,0,0,0,(-3*rho**0.5*(me["1p"] - 2*me["2p"]))/8,(-9*rho**0.5*(me["1p"] - 2*me["2p"]))/8,(-((1 + 2*rho)*me["1p"]) + 2*(2 + rho)*me["2p"])/8,(-3*(me["1p"] + 2*rho*me["1p"] - 2*(2 + rho)*me["2p"]))/8,(0.5 + rho)*me["1p"] - (-4 + rho)*me["2p"],(3*(me["1p"] + 2*rho*me["1p"] - 2*(-4 + rho)*me["2p"]))/2),
        (-(rho**0.5*(me["5"] + 2*(me["6"] + 3*me["7"] + 6*me["8"]))),-3*rho**0.5*(me["5"] + 2*me["6"]),6*(me["6"] + 6*me["8"]),18*me["6"],0,0,0,0,0,0,0,0,0,0,(rho**0.5*(me["1p"] + 2*(me["2p"] + 3*me["3p"] + 6*me["4p"])))/2,(3*rho**0.5*(me["1p"] + 2*me["2p"]))/2,(me["1p"] + 2*rho*me["1p"] - 2*(-4 + rho)*me["2p"] + 6*(me["3p"] + 2*rho*me["3p"] + 8*me["4p"] - 2*rho*me["4p"]))/6,(0.5 + rho)*me["1p"] - (-4 + rho)*me["2p"],(-2*(me["1p"] + 2*rho*me["1p"] - 2*(14 + rho)*me["2p"] + 6*(1 + 2*rho)*me["3p"] - 12*(14 + rho)*me["4p"]))/3,-2*(1 + 2*rho)*me["1p"] + 4*(14 + rho)*me["2p"]),
        (-3*rho**0.5*(me["5"] + 2*me["6"]),-9*rho**0.5*(me["5"] + 2*me["6"]),18*me["6"],54*me["6"],0,0,0,0,0,0,0,0,0,0,(3*rho**0.5*(me["1p"] + 2*me["2p"]))/2,(9*rho**0.5*(me["1p"] + 2*me["2p"]))/2,(0.5 + rho)*me["1p"] - (-4 + rho)*me["2p"],(3*(me["1p"] + 2*rho*me["1p"] - 2*(-4 + rho)*me["2p"]))/2,-2*(1 + 2*rho)*me["1p"] + 4*(14 + rho)*me["2p"],-6*(me["1p"] + 2*rho*me["1p"] - 2*(14 + rho)*me["2p"]))
    ))

    flavio.citations.register("Jager:2017gal")
    flavio.citations.register("Jager:2019bgk")
    # See eqs A.2 - A.7 in Lenz:2022pgw
    A_WE_cc = np.array((
        (((-1 + rho)*me["1"] + me["2"] + 2*rho*me["2"] + 6*((-1 + rho)*me["3"] + me["4"] + 2*rho*me["4"]))/3, (-1 + rho)*me["1"] + me["2"] + 2*rho*me["2"], -(rho**0.5*(me["2"] + 6*me["4"])), -3*rho**0.5*me["2"], (rho*(me["1"] + 6*me["3"]))/2, (3*rho*me["1"])/2, (rho**0.5*(me["1"] + 6*me["3"]))/4, (3*rho**0.5*me["1"])/4, -(rho**0.5*(me["1"] - 4*me["2"] + 6*(me["3"] - 4*me["4"]))), -3*rho**0.5*(me["1"] - 4*me["2"]), -(rho*(me["5p"] + 6*me["7p"])), -3*rho*me["5p"], rho**0.5*(me["6p"] + 6*me["8p"]), 3*rho**0.5*me["6p"], (me["5p"] - rho*me["5p"] - (1 + 2*rho)*me["6p"] - 6*((-1 + rho)*me["7p"] + me["8p"] + 2*rho*me["8p"]))/6, (me["5p"] - rho*me["5p"] - (1 + 2*rho)*me["6p"])/2, (rho**0.5*(me["5p"] - 2*(me["6p"] - 3*me["7p"] + 6*me["8p"])))/4, (3*rho**0.5*(me["5p"] - 2*me["6p"]))/4, -(rho**0.5*(me["5p"] + 2*(me["6p"] + 3*me["7p"] + 6*me["8p"]))), -3*rho**0.5*(me["5p"] + 2*me["6p"])),
        ((-1 + rho)*me["1"] + me["2"] + 2*rho*me["2"], 3*((-1 + rho)*me["1"] + me["2"] + 2*rho*me["2"]), -3*rho**0.5*me["2"], -9*rho**0.5*me["2"], (3*rho*me["1"])/2, (9*rho*me["1"])/2, (3*rho**0.5*me["1"])/4, (9*rho**0.5*me["1"])/4, -3*rho**0.5*(me["1"] - 4*me["2"]), -9*rho**0.5*(me["1"] - 4*me["2"]), -3*rho*me["5p"], -9*rho*me["5p"], 3*rho**0.5*me["6p"], 9*rho**0.5*me["6p"], (me["5p"] - rho*me["5p"] - (1 + 2*rho)*me["6p"])/2, (-3*((-1 + rho)*me["5p"] + me["6p"] + 2*rho*me["6p"]))/2, (3*rho**0.5*(me["5p"] - 2*me["6p"]))/4, (9*rho**0.5*(me["5p"] - 2*me["6p"]))/4, -3*rho**0.5*(me["5p"] + 2*me["6p"]), -9*rho**0.5*(me["5p"] + 2*me["6p"])),
        (-(rho**0.5*(me["2"] + 6*me["4"])), -3*rho**0.5*me["2"], -2*(-1 + 2*rho)*(me["2"] + 6*me["4"]), -6*(-1 + 2*rho)*me["2"], -(rho**0.5*(me["2"] + 6*me["4"]))/2, (-3*rho**0.5*me["2"])/2, -(rho*(me["2"] + 6*me["4"])), -3*rho*me["2"], -12*rho*(me["2"] + 6*me["4"]), -36*rho*me["2"], rho**0.5*(me["6p"] + 6*me["8p"]), 3*rho**0.5*me["6p"], -4*rho*(me["6p"] + 6*me["8p"]), -12*rho*me["6p"], (rho**0.5*(me["6p"] + 6*me["8p"]))/2, (3*rho**0.5*me["6p"])/2, -((-1 + 2*rho)*(me["6p"] + 6*me["8p"]))/2, (1.5 - 3*rho)*me["6p"], -6*(-1 + 2*rho)*(me["6p"] + 6*me["8p"]), -18*(-1 + 2*rho)*me["6p"]),
        (-3*rho**0.5*me["2"], -9*rho**0.5*me["2"], -6*(-1 + 2*rho)*me["2"], -18*(-1 + 2*rho)*me["2"], (-3*rho**0.5*me["2"])/2, (-9*rho**0.5*me["2"])/2, -3*rho*me["2"], -9*rho*me["2"], -36*rho*me["2"], -108*rho*me["2"], 3*rho**0.5*me["6p"], 9*rho**0.5*me["6p"], -12*rho*me["6p"], -36*rho*me["6p"], (3*rho**0.5*me["6p"])/2, (9*rho**0.5*me["6p"])/2, (1.5 - 3*rho)*me["6p"], (4.5 - 9*rho)*me["6p"], -18*(-1 + 2*rho)*me["6p"], -54*(-1 + 2*rho)*me["6p"]), 
        ((rho*(me["1"] + 6*me["3"]))/2, (3*rho*me["1"])/2, -(rho**0.5*(me["2"] + 6*me["4"]))/2, (-3*rho**0.5*me["2"])/2, ((-1 + rho)*me["1"] + me["2"] + 2*rho*me["2"] + 6*((-1 + rho)*me["3"] + me["4"] + 2*rho*me["4"]))/12, ((-1 + rho)*me["1"] + me["2"] + 2*rho*me["2"])/4, -(rho**0.5*(me["1"] - 2*(me["2"] - 3*me["3"] + 6*me["4"])))/8, (-3*rho**0.5*(me["1"] - 2*me["2"]))/8, (rho**0.5*(me["1"] + 2*(me["2"] + 3*me["3"] + 6*me["4"])))/2, (3*rho**0.5*(me["1"] + 2*me["2"]))/2, (me["5p"] - rho*me["5p"] - (1 + 2*rho)*me["6p"] - 6*((-1 + rho)*me["7p"] + me["8p"] + 2*rho*me["8p"]))/6, (me["5p"] - rho*me["5p"] - (1 + 2*rho)*me["6p"])/2, (rho**0.5*(me["6p"] + 6*me["8p"]))/2, (3*rho**0.5*me["6p"])/2, -(rho*(me["5p"] + 6*me["7p"]))/4, (-3*rho*me["5p"])/4, -(rho**0.5*(me["5p"] + 6*me["7p"]))/8, (-3*rho**0.5*me["5p"])/8, (rho**0.5*(me["5p"] - 4*me["6p"] + 6*(me["7p"] - 4*me["8p"])))/2, (3*rho**0.5*(me["5p"] - 4*me["6p"]))/2),
        ((3*rho*me["1"])/2, (9*rho*me["1"])/2, (-3*rho**0.5*me["2"])/2, (-9*rho**0.5*me["2"])/2, ((-1 + rho)*me["1"] + me["2"] + 2*rho*me["2"])/4, (3*((-1 + rho)*me["1"] + me["2"] + 2*rho*me["2"]))/4, (-3*rho**0.5*(me["1"] - 2*me["2"]))/8, (-9*rho**0.5*(me["1"] - 2*me["2"]))/8, (3*rho**0.5*(me["1"] + 2*me["2"]))/2, (9*rho**0.5*(me["1"] + 2*me["2"]))/2, (me["5p"] - rho*me["5p"] - (1 + 2*rho)*me["6p"])/2, (-3*((-1 + rho)*me["5p"] + me["6p"] + 2*rho*me["6p"]))/2, (3*rho**0.5*me["6p"])/2, (9*rho**0.5*me["6p"])/2, (-3*rho*me["5p"])/4, (-9*rho*me["5p"])/4, (-3*rho**0.5*me["5p"])/8, (-9*rho**0.5*me["5p"])/8, (3*rho**0.5*(me["5p"] - 4*me["6p"]))/2, (9*rho**0.5*(me["5p"] - 4*me["6p"]))/2),
        ((rho**0.5*(me["1"] + 6*me["3"]))/4, (3*rho**0.5*me["1"])/4, -(rho*(me["2"] + 6*me["4"])), -3*rho*me["2"], -(rho**0.5*(me["1"] - 2*(me["2"] - 3*me["3"] + 6*me["4"])))/8, (-3*rho**0.5*(me["1"] - 2*me["2"]))/8, (-((1 + 2*rho)*me["1"]) - 4*(-1 + rho)*me["2"] - 6*(me["3"] + 2*rho*me["3"] + 4*(-1 + rho)*me["4"]))/24, (-((1 + 2*rho)*me["1"]) - 4*(-1 + rho)*me["2"])/8, (me["1"] + 2*rho*me["1"] + (8 - 20*rho)*me["2"] + 6*(me["3"] + 2*rho*me["3"] + 8*me["4"] - 20*rho*me["4"]))/6, (0.5 + rho)*me["1"] + 2*(2 - 5*rho)*me["2"], (rho**0.5*(me["5p"] - 2*(me["6p"] - 3*me["7p"] + 6*me["8p"])))/4, (3*rho**0.5*(me["5p"] - 2*me["6p"]))/4, -((-1 + 2*rho)*(me["6p"] + 6*me["8p"]))/2, (1.5 - 3*rho)*me["6p"], -(rho**0.5*(me["5p"] + 6*me["7p"]))/8, (-3*rho**0.5*me["5p"])/8, -(rho*(me["5p"] + 6*me["7p"]))/4, (-3*rho*me["5p"])/4, rho*(me["5p"] - 4*me["6p"] + 6*(me["7p"] - 4*me["8p"])), 3*rho*(me["5p"] - 4*me["6p"])),
        ((3*rho**0.5*me["1"])/4, (9*rho**0.5*me["1"])/4, -3*rho*me["2"], -9*rho*me["2"], (-3*rho**0.5*(me["1"] - 2*me["2"]))/8, (-9*rho**0.5*(me["1"] - 2*me["2"]))/8, (-((1 + 2*rho)*me["1"]) - 4*(-1 + rho)*me["2"])/8, (-3*(me["1"] + 2*rho*me["1"] + 4*(-1 + rho)*me["2"]))/8, (0.5 + rho)*me["1"] + 2*(2 - 5*rho)*me["2"], (3*(me["1"] + 2*rho*me["1"] + 4*(2 - 5*rho)*me["2"]))/2, (3*rho**0.5*(me["5p"] - 2*me["6p"]))/4, (9*rho**0.5*(me["5p"] - 2*me["6p"]))/4, (1.5 - 3*rho)*me["6p"], (4.5 - 9*rho)*me["6p"], (-3*rho**0.5*me["5p"])/8, (-9*rho**0.5*me["5p"])/8, (-3*rho*me["5p"])/4, (-9*rho*me["5p"])/4, 3*rho*(me["5p"] - 4*me["6p"]), 9*rho*(me["5p"] - 4*me["6p"])),
        (-(rho**0.5*(me["1"] - 4*me["2"] + 6*(me["3"] - 4*me["4"]))), -3*rho**0.5*(me["1"] - 4*me["2"]), -12*rho*(me["2"] + 6*me["4"]), -36*rho*me["2"], (rho**0.5*(me["1"] + 2*(me["2"] + 3*me["3"] + 6*me["4"])))/2, (3*rho**0.5*(me["1"] + 2*me["2"]))/2, (me["1"] + 2*rho*me["1"] + (8 - 20*rho)*me["2"] + 6*(me["3"] + 2*rho*me["3"] + 8*me["4"] - 20*rho*me["4"]))/6, (0.5 + rho)*me["1"] + 2*(2 - 5*rho)*me["2"], (-2*(me["1"] + 2*rho*me["1"] + 4*(-7 + 13*rho)*me["2"] + 6*(me["3"] + 2*rho*me["3"] - 28*me["4"] + 52*rho*me["4"])))/3, -2*(me["1"] + 2*rho*me["1"] + 4*(-7 + 13*rho)*me["2"]), -(rho**0.5*(me["5p"] + 2*(me["6p"] + 3*me["7p"] + 6*me["8p"]))), -3*rho**0.5*(me["5p"] + 2*me["6p"]), -6*(-1 + 2*rho)*(me["6p"] + 6*me["8p"]), -18*(-1 + 2*rho)*me["6p"], (rho**0.5*(me["5p"] - 4*me["6p"] + 6*(me["7p"] - 4*me["8p"])))/2, (3*rho**0.5*(me["5p"] - 4*me["6p"]))/2, rho*(me["5p"] - 4*me["6p"] + 6*(me["7p"] - 4*me["8p"])), 3*rho*(me["5p"] - 4*me["6p"]), -4*rho*(me["5p"] + 8*me["6p"] + 6*(me["7p"] + 8*me["8p"])), -12*rho*(me["5p"] + 8*me["6p"])),
        (-3*rho**0.5*(me["1"] - 4*me["2"]), -9*rho**0.5*(me["1"] - 4*me["2"]), -36*rho*me["2"], -108*rho*me["2"], (3*rho**0.5*(me["1"] + 2*me["2"]))/2, (9*rho**0.5*(me["1"] + 2*me["2"]))/2, (0.5 + rho)*me["1"] + 2*(2 - 5*rho)*me["2"], (3*(me["1"] + 2*rho*me["1"] + 4*(2 - 5*rho)*me["2"]))/2, -2*(me["1"] + 2*rho*me["1"] + 4*(-7 + 13*rho)*me["2"]), -6*(me["1"] + 2*rho*me["1"] + 4*(-7 + 13*rho)*me["2"]), -3*rho**0.5*(me["5p"] + 2*me["6p"]), -9*rho**0.5*(me["5p"] + 2*me["6p"]), -18*(-1 + 2*rho)*me["6p"], -54*(-1 + 2*rho)*me["6p"], (3*rho**0.5*(me["5p"] - 4*me["6p"]))/2, (9*rho**0.5*(me["5p"] - 4*me["6p"]))/2, 3*rho*(me["5p"] - 4*me["6p"]), 9*rho*(me["5p"] - 4*me["6p"]), -12*rho*(me["5p"] + 8*me["6p"]), -36*rho*(me["5p"] + 8*me["6p"])),
        (-(rho*(me["5"] + 6*me["7"])), -3*rho*me["5"], rho**0.5*(me["6"] + 6*me["8"]), 3*rho**0.5*me["6"], (me["5"] - rho*me["5"] - (1 + 2*rho)*me["6"] - 6*((-1 + rho)*me["7"] + me["8"] + 2*rho*me["8"]))/6, (me["5"] - rho*me["5"] - (1 + 2*rho)*me["6"])/2, (rho**0.5*(me["5"] - 2*(me["6"] - 3*me["7"] + 6*me["8"])))/4, (3*rho**0.5*(me["5"] - 2*me["6"]))/4, -(rho**0.5*(me["5"] + 2*(me["6"] + 3*me["7"] + 6*me["8"]))), -3*rho**0.5*(me["5"] + 2*me["6"]), ((-1 + rho)*me["1p"] + me["2p"] + 2*rho*me["2p"] + 6*((-1 + rho)*me["3p"] + me["4p"] + 2*rho*me["4p"]))/3, (-1 + rho)*me["1p"] + me["2p"] + 2*rho*me["2p"], -(rho**0.5*(me["2p"] + 6*me["4p"])), -3*rho**0.5*me["2p"], (rho*(me["1p"] + 6*me["3p"]))/2, (3*rho*me["1p"])/2, (rho**0.5*(me["1p"] + 6*me["3p"]))/4, (3*rho**0.5*me["1p"])/4, -(rho**0.5*(me["1p"] - 4*me["2p"] + 6*(me["3p"] - 4*me["4p"]))), -3*rho**0.5*(me["1p"] - 4*me["2p"])),
        (-3*rho*me["5"], -9*rho*me["5"], 3*rho**0.5*me["6"], 9*rho**0.5*me["6"], (me["5"] - rho*me["5"] - (1 + 2*rho)*me["6"])/2, (-3*((-1 + rho)*me["5"] + me["6"] + 2*rho*me["6"]))/2, (3*rho**0.5*(me["5"] - 2*me["6"]))/4, (9*rho**0.5*(me["5"] - 2*me["6"]))/4, -3*rho**0.5*(me["5"] + 2*me["6"]), -9*rho**0.5*(me["5"] + 2*me["6"]), (-1 + rho)*me["1p"] + me["2p"] + 2*rho*me["2p"], 3*((-1 + rho)*me["1p"] + me["2p"] + 2*rho*me["2p"]), -3*rho**0.5*me["2p"], -9*rho**0.5*me["2p"], (3*rho*me["1p"])/2, (9*rho*me["1p"])/2, (3*rho**0.5*me["1p"])/4, (9*rho**0.5*me["1p"])/4, -3*rho**0.5*(me["1p"] - 4*me["2p"]), -9*rho**0.5*(me["1p"] - 4*me["2p"])),
        (rho**0.5*(me["6"] + 6*me["8"]), 3*rho**0.5*me["6"], -4*rho*(me["6"] + 6*me["8"]), -12*rho*me["6"], (rho**0.5*(me["6"] + 6*me["8"]))/2, (3*rho**0.5*me["6"])/2, -((-1 + 2*rho)*(me["6"] + 6*me["8"]))/2, (1.5 - 3*rho)*me["6"], -6*(-1 + 2*rho)*(me["6"] + 6*me["8"]), -18*(-1 + 2*rho)*me["6"], -(rho**0.5*(me["2p"] + 6*me["4p"])), -3*rho**0.5*me["2p"], -2*(-1 + 2*rho)*(me["2p"] + 6*me["4p"]), -6*(-1 + 2*rho)*me["2p"], -(rho**0.5*(me["2p"] + 6*me["4p"]))/2, (-3*rho**0.5*me["2p"])/2, -(rho*(me["2p"] + 6*me["4p"])), -3*rho*me["2p"], -12*rho*(me["2p"] + 6*me["4p"]), -36*rho*me["2p"]),
        (3*rho**0.5*me["6"], 9*rho**0.5*me["6"], -12*rho*me["6"], -36*rho*me["6"], (3*rho**0.5*me["6"])/2, (9*rho**0.5*me["6"])/2, (1.5 - 3*rho)*me["6"], (4.5 - 9*rho)*me["6"], -18*(-1 + 2*rho)*me["6"], -54*(-1 + 2*rho)*me["6"], -3*rho**0.5*me["2p"], -9*rho**0.5*me["2p"], -6*(-1 + 2*rho)*me["2p"], -18*(-1 + 2*rho)*me["2p"], (-3*rho**0.5*me["2p"])/2, (-9*rho**0.5*me["2p"])/2, -3*rho*me["2p"], -9*rho*me["2p"], -36*rho*me["2p"], -108*rho*me["2p"]),
        ((me["5"] - rho*me["5"] - (1 + 2*rho)*me["6"] - 6*((-1 + rho)*me["7"] + me["8"] + 2*rho*me["8"]))/6, (me["5"] - rho*me["5"] - (1 + 2*rho)*me["6"])/2, (rho**0.5*(me["6"] + 6*me["8"]))/2, (3*rho**0.5*me["6"])/2, -(rho*(me["5"] + 6*me["7"]))/4, (-3*rho*me["5"])/4, -(rho**0.5*(me["5"] + 6*me["7"]))/8, (-3*rho**0.5*me["5"])/8, (rho**0.5*(me["5"] - 4*me["6"] + 6*(me["7"] - 4*me["8"])))/2, (3*rho**0.5*(me["5"] - 4*me["6"]))/2, (rho*(me["1p"] + 6*me["3p"]))/2, (3*rho*me["1p"])/2, -(rho**0.5*(me["2p"] + 6*me["4p"]))/2, (-3*rho**0.5*me["2p"])/2, ((-1 + rho)*me["1p"] + me["2p"] + 2*rho*me["2p"] + 6*((-1 + rho)*me["3p"] + me["4p"] + 2*rho*me["4p"]))/12, ((-1 + rho)*me["1p"] + me["2p"] + 2*rho*me["2p"])/4, -(rho**0.5*(me["1p"] - 2*(me["2p"] - 3*me["3p"] + 6*me["4p"])))/8, (-3*rho**0.5*(me["1p"] - 2*me["2p"]))/8, (rho**0.5*(me["1p"] + 2*(me["2p"] + 3*me["3p"] + 6*me["4p"])))/2, (3*rho**0.5*(me["1p"] + 2*me["2p"]))/2),
        ((me["5"] - rho*me["5"] - (1 + 2*rho)*me["6"])/2, (-3*((-1 + rho)*me["5"] + me["6"] + 2*rho*me["6"]))/2, (3*rho**0.5*me["6"])/2, (9*rho**0.5*me["6"])/2, (-3*rho*me["5"])/4, (-9*rho*me["5"])/4, (-3*rho**0.5*me["5"])/8, (-9*rho**0.5*me["5"])/8, (3*rho**0.5*(me["5"] - 4*me["6"]))/2, (9*rho**0.5*(me["5"] - 4*me["6"]))/2, (3*rho*me["1p"])/2, (9*rho*me["1p"])/2, (-3*rho**0.5*me["2p"])/2, (-9*rho**0.5*me["2p"])/2, ((-1 + rho)*me["1p"] + me["2p"] + 2*rho*me["2p"])/4, (3*((-1 + rho)*me["1p"] + me["2p"] + 2*rho*me["2p"]))/4, (-3*rho**0.5*(me["1p"] - 2*me["2p"]))/8, (-9*rho**0.5*(me["1p"] - 2*me["2p"]))/8, (3*rho**0.5*(me["1p"] + 2*me["2p"]))/2, (9*rho**0.5*(me["1p"] + 2*me["2p"]))/2),
        ((rho**0.5*(me["5"] - 2*(me["6"] - 3*me["7"] + 6*me["8"])))/4, (3*rho**0.5*(me["5"] - 2*me["6"]))/4, -((-1 + 2*rho)*(me["6"] + 6*me["8"]))/2, (1.5 - 3*rho)*me["6"], -(rho**0.5*(me["5"] + 6*me["7"]))/8, (-3*rho**0.5*me["5"])/8, -(rho*(me["5"] + 6*me["7"]))/4, (-3*rho*me["5"])/4, rho*(me["5"] - 4*me["6"] + 6*(me["7"] - 4*me["8"])), 3*rho*(me["5"] - 4*me["6"]), (rho**0.5*(me["1p"] + 6*me["3p"]))/4, (3*rho**0.5*me["1p"])/4, -(rho*(me["2p"] + 6*me["4p"])), -3*rho*me["2p"], -(rho**0.5*(me["1p"] - 2*(me["2p"] - 3*me["3p"] + 6*me["4p"])))/8, (-3*rho**0.5*(me["1p"] - 2*me["2p"]))/8, (-((1 + 2*rho)*me["1p"]) - 4*(-1 + rho)*me["2p"] - 6*(me["3p"] + 2*rho*me["3p"] + 4*(-1 + rho)*me["4p"]))/24, (-((1 + 2*rho)*me["1p"]) - 4*(-1 + rho)*me["2p"])/8, (me["1p"] + 2*rho*me["1p"] + (8 - 20*rho)*me["2p"] + 6*(me["3p"] + 2*rho*me["3p"] + 8*me["4p"] - 20*rho*me["4p"]))/6, (0.5 + rho)*me["1p"] + 2*(2 - 5*rho)*me["2p"]),
        ((3*rho**0.5*(me["5"] - 2*me["6"]))/4, (9*rho**0.5*(me["5"] - 2*me["6"]))/4, (1.5 - 3*rho)*me["6"], (4.5 - 9*rho)*me["6"], (-3*rho**0.5*me["5"])/8, (-9*rho**0.5*me["5"])/8, (-3*rho*me["5"])/4, (-9*rho*me["5"])/4, 3*rho*(me["5"] - 4*me["6"]), 9*rho*(me["5"] - 4*me["6"]), (3*rho**0.5*me["1p"])/4, (9*rho**0.5*me["1p"])/4, -3*rho*me["2p"], -9*rho*me["2p"], (-3*rho**0.5*(me["1p"] - 2*me["2p"]))/8, (-9*rho**0.5*(me["1p"] - 2*me["2p"]))/8, (-((1 + 2*rho)*me["1p"]) - 4*(-1 + rho)*me["2p"])/8, (-3*(me["1p"] + 2*rho*me["1p"] + 4*(-1 + rho)*me["2p"]))/8, (0.5 + rho)*me["1p"] + 2*(2 - 5*rho)*me["2p"], (3*(me["1p"] + 2*rho*me["1p"] + 4*(2 - 5*rho)*me["2p"]))/2),
        (-(rho**0.5*(me["5"] + 2*(me["6"] + 3*me["7"] + 6*me["8"]))), -3*rho**0.5*(me["5"] + 2*me["6"]), -6*(-1 + 2*rho)*(me["6"] + 6*me["8"]), -18*(-1 + 2*rho)*me["6"], (rho**0.5*(me["5"] - 4*me["6"] + 6*(me["7"] - 4*me["8"])))/2, (3*rho**0.5*(me["5"] - 4*me["6"]))/2, rho*(me["5"] - 4*me["6"] + 6*(me["7"] - 4*me["8"])), 3*rho*(me["5"] - 4*me["6"]), -4*rho*(me["5"] + 8*me["6"] + 6*(me["7"] + 8*me["8"])), -12*rho*(me["5"] + 8*me["6"]), -(rho**0.5*(me["1p"] - 4*me["2p"] + 6*(me["3p"] - 4*me["4p"]))), -3*rho**0.5*(me["1p"] - 4*me["2p"]), -12*rho*(me["2p"] + 6*me["4p"]), -36*rho*me["2p"], (rho**0.5*(me["1p"] + 2*(me["2p"] + 3*me["3p"] + 6*me["4p"])))/2, (3*rho**0.5*(me["1p"] + 2*me["2p"]))/2, (me["1p"] + 2*rho*me["1p"] + (8 - 20*rho)*me["2p"] + 6*(me["3p"] + 2*rho*me["3p"] + 8*me["4p"] - 20*rho*me["4p"]))/6, (0.5 + rho)*me["1p"] + 2*(2 - 5*rho)*me["2p"], (-2*(me["1p"] + 2*rho*me["1p"] + 4*(-7 + 13*rho)*me["2p"] + 6*(me["3p"] + 2*rho*me["3p"] - 28*me["4p"] + 52*rho*me["4p"])))/3, -2*(me["1p"] + 2*rho*me["1p"] + 4*(-7 + 13*rho)*me["2p"])),
        (-3*rho**0.5*(me["5"] + 2*me["6"]), -9*rho**0.5*(me["5"] + 2*me["6"]), -18*(-1 + 2*rho)*me["6"], -54*(-1 + 2*rho)*me["6"], (3*rho**0.5*(me["5"] - 4*me["6"]))/2, (9*rho**0.5*(me["5"] - 4*me["6"]))/2, 3*rho*(me["5"] - 4*me["6"]), 9*rho*(me["5"] - 4*me["6"]), -12*rho*(me["5"] + 8*me["6"]), -36*rho*(me["5"] + 8*me["6"]), -3*rho**0.5*(me["1p"] - 4*me["2p"]), -9*rho**0.5*(me["1p"] - 4*me["2p"]), -36*rho*me["2p"], -108*rho*me["2p"], (3*rho**0.5*(me["1p"] + 2*me["2p"]))/2, (9*rho**0.5*(me["1p"] + 2*me["2p"]))/2, (0.5 + rho)*me["1p"] + 2*(2 - 5*rho)*me["2p"], (3*(me["1p"] + 2*rho*me["1p"] + 4*(2 - 5*rho)*me["2p"]))/2, -2*(me["1p"] + 2*rho*me["1p"] + 4*(-7 + 13*rho)*me["2p"]), -6*(me["1p"] + 2*rho*me["1p"] + 4*(-7 + 13*rho)*me["2p"]))
    ))

    dflav = {'d': 0, 's': 1, 'b': 2}
    uflav = {'u': 0, 'c': 1, 't': 2}
    result = 0
    for sector in ("dbcu", "dbcc"):
        CSM, CNP = siegen_basis_wcs(wc_obj, par, sector)
        C = CSM + CNP
        ckm_factor = V[uflav[sector[3]], dflav[sector[0]]] * V[uflav[sector[2]], dflav[sector[1]]] # Vud Vcb or Vcd Vcb
        if sector == "dbcu":
            phase_space = (1 - rho)**2
            A_WE = A_WE_cu
        elif sector == "dbcc":
            phase_space = (1 - 4*rho)**0.5
            A_WE = A_WE_cc
        prefactor = GF**2 * mb**2 * abs(ckm_factor)**2 * phase_space / (6 * pi)

        # While this term is manifestly real, we take take the real part to avoid
        # tiny imaginary parts from floating point errors
        result += prefactor * (C @ A_WE @ C.conj() - CSM @ A_WE @ CSM.conj()).real

    return result


def pauli_interference(wc_obj, par, meson):
    r"""BSM Pauli interference contributions. For now, only from b->c ubar d
    (dbcu sector) and b->c ubar s (sbcu sector) operators."""
    # For Bd, no PI from dbcu or sbcu sectors
    if meson == "B0":
        return 0
    # So now B+ case only: both dbcu and sbcu sectors contribute (with different CKM factors)

    GF = par["GF"]
    mb = flavio.physics.running.running.get_mb_KS(par, 1)
    mc = par["m_c"]
    rho = mc**2 / mb**2
    V = ckm.get_ckm(par)

    me = lifetimematrixelements(par, meson, config["renormalization scale"]["b lifetime ratios"])
    # Matrix for Pauli interference contributions with down and strange quark in the loop are the same
    flavio.citations.register("Lenz:2022pgw")
    # See eqs 2.22 - 2.27 in Lenz:2022pgw
    A_PI_cd = np.array((
        (me["1"] + 6*me["3"],3*me["1"],-(rho**0.5*(me["1"] + 6*me["3"]))/2,(-3*rho**0.5*me["1"])/2,-(rho**0.5*(me["5"] - 2*(me["6"] - 3*me["7"] + 6*me["8"])))/4,(-3*rho**0.5*(me["5"] - 2*me["6"]))/4,(-me["5"] + 2*(me["6"] - 3*me["7"] + 6*me["8"]))/4,(-3*(me["5"] - 2*me["6"]))/4,3*(me["5"] - 2*(me["6"] - 3*me["7"] + 6*me["8"])),9*(me["5"] - 2*me["6"]),0,0,0,0,0,0,0,0,0,0),
        (3*me["1"],me["1"] + 6*me["3"],(-3*rho**0.5*me["1"])/2,-(rho**0.5*(me["1"] + 6*me["3"]))/2,(-3*rho**0.5*(me["5"] - 2*me["6"]))/4,-(rho**0.5*(me["5"] - 2*(me["6"] - 3*me["7"] + 6*me["8"])))/4,(-3*(me["5"] - 2*me["6"]))/4,(-me["5"] + 2*(me["6"] - 3*me["7"] + 6*me["8"]))/4,9*(me["5"] - 2*me["6"]),3*(me["5"] - 2*(me["6"] - 3*me["7"] + 6*me["8"])),0,0,0,0,0,0,0,0,0,0),
        (-(rho**0.5*(me["1"] + 6*me["3"]))/2,(-3*rho**0.5*me["1"])/2,(me["1"] + 2*rho*me["1"] - 2*(2 + rho)*me["2"] + 6*(me["3"] + 2*rho*me["3"] - 2*(2 + rho)*me["4"]))/6,(0.5 + rho)*me["1"] - (2 + rho)*me["2"],((-1 + rho)*me["5"] - 2*(me["6"] + 2*rho*me["6"] + 3*(me["7"] - rho*me["7"] + 2*me["8"] + 4*rho*me["8"])))/12,((-1 + rho)*me["5"] - 2*(me["6"] + 2*rho*me["6"]))/4,-(rho**0.5*(me["6"] + 6*me["8"]))/2,(-3*rho**0.5*me["6"])/2,-2*rho**0.5*(me["5"] - me["6"] + 6*me["7"] - 6*me["8"]),-6*rho**0.5*(me["5"] - me["6"]),0,0,0,0,0,0,0,0,0,0),
        ((-3*rho**0.5*me["1"])/2,-(rho**0.5*(me["1"] + 6*me["3"]))/2,(0.5 + rho)*me["1"] - (2 + rho)*me["2"],(me["1"] + 2*rho*me["1"] - 2*(2 + rho)*me["2"] + 6*(me["3"] + 2*rho*me["3"] - 2*(2 + rho)*me["4"]))/6,((-1 + rho)*me["5"] - 2*(me["6"] + 2*rho*me["6"]))/4,((-1 + rho)*me["5"] - 2*(me["6"] + 2*rho*me["6"] + 3*(me["7"] - rho*me["7"] + 2*me["8"] + 4*rho*me["8"])))/12,(-3*rho**0.5*me["6"])/2,-(rho**0.5*(me["6"] + 6*me["8"]))/2,-6*rho**0.5*(me["5"] - me["6"]),-2*rho**0.5*(me["5"] - me["6"] + 6*me["7"] - 6*me["8"]),0,0,0,0,0,0,0,0,0,0),
        (-(rho**0.5*(me["5p"] - 2*(me["6p"] - 3*me["7p"] + 6*me["8p"])))/4,(-3*rho**0.5*(me["5p"] - 2*me["6p"]))/4,((-1 + rho)*me["5p"] - 2*(me["6p"] + 2*rho*me["6p"] + 3*(me["7p"] - rho*me["7p"] + 2*me["8p"] + 4*rho*me["8p"])))/12,((-1 + rho)*me["5p"] - 2*(me["6p"] + 2*rho*me["6p"]))/4,(me["1p"] + 2*rho*me["1p"] - 2*(2 + rho)*me["2p"] + 6*(me["3p"] + 2*rho*me["3p"] - 2*(2 + rho)*me["4p"]))/24,(me["1p"] + 2*rho*me["1p"] - 2*(2 + rho)*me["2p"])/8,(rho**0.5*(me["1p"] - 2*(me["2p"] - 3*me["3p"] + 6*me["4p"])))/8,(3*rho**0.5*(me["1p"] - 2*me["2p"]))/8,-(rho**0.5*(me["1p"] + 2*(me["2p"] + 3*me["3p"] + 6*me["4p"])))/2,(-3*rho**0.5*(me["1p"] + 2*me["2p"]))/2,0,0,0,0,0,0,0,0,0,0),
        ((-3*rho**0.5*(me["5p"] - 2*me["6p"]))/4,-(rho**0.5*(me["5p"] - 2*(me["6p"] - 3*me["7p"] + 6*me["8p"])))/4,((-1 + rho)*me["5p"] - 2*(me["6p"] + 2*rho*me["6p"]))/4,((-1 + rho)*me["5p"] - 2*(me["6p"] + 2*rho*me["6p"] + 3*(me["7p"] - rho*me["7p"] + 2*me["8p"] + 4*rho*me["8p"])))/12,(me["1p"] + 2*rho*me["1p"] - 2*(2 + rho)*me["2p"])/8,(me["1p"] + 2*rho*me["1p"] - 2*(2 + rho)*me["2p"] + 6*(me["3p"] + 2*rho*me["3p"] - 2*(2 + rho)*me["4p"]))/24,(3*rho**0.5*(me["1p"] - 2*me["2p"]))/8,(rho**0.5*(me["1p"] - 2*(me["2p"] - 3*me["3p"] + 6*me["4p"])))/8,(-3*rho**0.5*(me["1p"] + 2*me["2p"]))/2,-(rho**0.5*(me["1p"] + 2*(me["2p"] + 3*me["3p"] + 6*me["4p"])))/2,0,0,0,0,0,0,0,0,0,0),
        ((-me["5p"] + 2*(me["6p"] - 3*me["7p"] + 6*me["8p"]))/4,(-3*(me["5p"] - 2*me["6p"]))/4,-(rho**0.5*(me["6p"] + 6*me["8p"]))/2,(-3*rho**0.5*me["6p"])/2,(rho**0.5*(me["1p"] - 2*(me["2p"] - 3*me["3p"] + 6*me["4p"])))/8,(3*rho**0.5*(me["1p"] - 2*me["2p"]))/8,((2 + rho)*me["1p"] - 2*(me["2p"] + 2*rho*me["2p"] - 3*(2 + rho)*me["3p"] + 6*(me["4p"] + 2*rho*me["4p"])))/24,((2 + rho)*me["1p"] - 2*(me["2p"] + 2*rho*me["2p"]))/8,((-4 + rho)*me["1p"] - 2*(me["2p"] + 2*rho*me["2p"] - 3*(-4 + rho)*me["3p"] + 6*(me["4p"] + 2*rho*me["4p"])))/6,((-4 + rho)*me["1p"])/2 - (1 + 2*rho)*me["2p"],0,0,0,0,0,0,0,0,0,0),
        ((-3*(me["5p"] - 2*me["6p"]))/4,(-me["5p"] + 2*(me["6p"] - 3*me["7p"] + 6*me["8p"]))/4,(-3*rho**0.5*me["6p"])/2,-(rho**0.5*(me["6p"] + 6*me["8p"]))/2,(3*rho**0.5*(me["1p"] - 2*me["2p"]))/8,(rho**0.5*(me["1p"] - 2*(me["2p"] - 3*me["3p"] + 6*me["4p"])))/8,((2 + rho)*me["1p"] - 2*(me["2p"] + 2*rho*me["2p"]))/8,((2 + rho)*me["1p"] - 2*(me["2p"] + 2*rho*me["2p"] - 3*(2 + rho)*me["3p"] + 6*(me["4p"] + 2*rho*me["4p"])))/24,((-4 + rho)*me["1p"])/2 - (1 + 2*rho)*me["2p"],((-4 + rho)*me["1p"] - 2*(me["2p"] + 2*rho*me["2p"] - 3*(-4 + rho)*me["3p"] + 6*(me["4p"] + 2*rho*me["4p"])))/6,0,0,0,0,0,0,0,0,0,0),
        (3*(me["5p"] - 2*(me["6p"] - 3*me["7p"] + 6*me["8p"])),9*(me["5p"] - 2*me["6p"]),-2*rho**0.5*(me["5p"] - me["6p"] + 6*me["7p"] - 6*me["8p"]),-6*rho**0.5*(me["5p"] - me["6p"]),-(rho**0.5*(me["1p"] + 2*(me["2p"] + 3*me["3p"] + 6*me["4p"])))/2,(-3*rho**0.5*(me["1p"] + 2*me["2p"]))/2,((-4 + rho)*me["1p"] - 2*(me["2p"] + 2*rho*me["2p"] - 3*(-4 + rho)*me["3p"] + 6*(me["4p"] + 2*rho*me["4p"])))/6,((-4 + rho)*me["1p"])/2 - (1 + 2*rho)*me["2p"],(2*((14 + rho)*me["1p"] - 2*(me["2p"] + 2*rho*me["2p"] - 3*(14 + rho)*me["3p"] + 6*(me["4p"] + 2*rho*me["4p"]))))/3,2*(14 + rho)*me["1p"] - 4*(me["2p"] + 2*rho*me["2p"]),0,0,0,0,0,0,0,0,0,0),
        (9*(me["5p"] - 2*me["6p"]),3*(me["5p"] - 2*(me["6p"] - 3*me["7p"] + 6*me["8p"])),-6*rho**0.5*(me["5p"] - me["6p"]),-2*rho**0.5*(me["5p"] - me["6p"] + 6*me["7p"] - 6*me["8p"]),(-3*rho**0.5*(me["1p"] + 2*me["2p"]))/2,-(rho**0.5*(me["1p"] + 2*(me["2p"] + 3*me["3p"] + 6*me["4p"])))/2,((-4 + rho)*me["1p"])/2 - (1 + 2*rho)*me["2p"],((-4 + rho)*me["1p"] - 2*(me["2p"] + 2*rho*me["2p"] - 3*(-4 + rho)*me["3p"] + 6*(me["4p"] + 2*rho*me["4p"])))/6,2*(14 + rho)*me["1p"] - 4*(me["2p"] + 2*rho*me["2p"]),(2*((14 + rho)*me["1p"] - 2*(me["2p"] + 2*rho*me["2p"] - 3*(14 + rho)*me["3p"] + 6*(me["4p"] + 2*rho*me["4p"]))))/3,0,0,0,0,0,0,0,0,0,0),
        (0,0,0,0,0,0,0,0,0,0,me["1p"] + 6*me["3p"],3*me["1p"],-(rho**0.5*(me["1p"] + 6*me["3p"]))/2,(-3*rho**0.5*me["1p"])/2,-(rho**0.5*(me["5p"] - 2*(me["6p"] - 3*me["7p"] + 6*me["8p"])))/4,(-3*rho**0.5*(me["5p"] - 2*me["6p"]))/4,(-me["5p"] + 2*(me["6p"] - 3*me["7p"] + 6*me["8p"]))/4,(-3*(me["5p"] - 2*me["6p"]))/4,3*(me["5p"] - 2*(me["6p"] - 3*me["7p"] + 6*me["8p"])),9*(me["5p"] - 2*me["6p"])),
        (0,0,0,0,0,0,0,0,0,0,3*me["1p"],me["1p"] + 6*me["3p"],(-3*rho**0.5*me["1p"])/2,-(rho**0.5*(me["1p"] + 6*me["3p"]))/2,(-3*rho**0.5*(me["5p"] - 2*me["6p"]))/4,-(rho**0.5*(me["5p"] - 2*(me["6p"] - 3*me["7p"] + 6*me["8p"])))/4,(-3*(me["5p"] - 2*me["6p"]))/4,(-me["5p"] + 2*(me["6p"] - 3*me["7p"] + 6*me["8p"]))/4,9*(me["5p"] - 2*me["6p"]),3*(me["5p"] - 2*(me["6p"] - 3*me["7p"] + 6*me["8p"]))),
        (0,0,0,0,0,0,0,0,0,0,-(rho**0.5*(me["1p"] + 6*me["3p"]))/2,(-3*rho**0.5*me["1p"])/2,(me["1p"] + 2*rho*me["1p"] - 2*(2 + rho)*me["2p"] + 6*(me["3p"] + 2*rho*me["3p"] - 2*(2 + rho)*me["4p"]))/6,(0.5 + rho)*me["1p"] - (2 + rho)*me["2p"],((-1 + rho)*me["5p"] - 2*(me["6p"] + 2*rho*me["6p"] + 3*(me["7p"] - rho*me["7p"] + 2*me["8p"] + 4*rho*me["8p"])))/12,((-1 + rho)*me["5p"] - 2*(me["6p"] + 2*rho*me["6p"]))/4,-(rho**0.5*(me["6p"] + 6*me["8p"]))/2,(-3*rho**0.5*me["6p"])/2,-2*rho**0.5*(me["5p"] - me["6p"] + 6*me["7p"] - 6*me["8p"]),-6*rho**0.5*(me["5p"] - me["6p"])),
        (0,0,0,0,0,0,0,0,0,0,(-3*rho**0.5*me["1p"])/2,-(rho**0.5*(me["1p"] + 6*me["3p"]))/2,(0.5 + rho)*me["1p"] - (2 + rho)*me["2p"],(me["1p"] + 2*rho*me["1p"] - 2*(2 + rho)*me["2p"] + 6*(me["3p"] + 2*rho*me["3p"] - 2*(2 + rho)*me["4p"]))/6,((-1 + rho)*me["5p"] - 2*(me["6p"] + 2*rho*me["6p"]))/4,((-1 + rho)*me["5p"] - 2*(me["6p"] + 2*rho*me["6p"] + 3*(me["7p"] - rho*me["7p"] + 2*me["8p"] + 4*rho*me["8p"])))/12,(-3*rho**0.5*me["6p"])/2,-(rho**0.5*(me["6p"] + 6*me["8p"]))/2,-6*rho**0.5*(me["5p"] - me["6p"]),-2*rho**0.5*(me["5p"] - me["6p"] + 6*me["7p"] - 6*me["8p"])),
        (0,0,0,0,0,0,0,0,0,0,-(rho**0.5*(me["5"] - 2*(me["6"] - 3*me["7"] + 6*me["8"])))/4,(-3*rho**0.5*(me["5"] - 2*me["6"]))/4,((-1 + rho)*me["5"] - 2*(me["6"] + 2*rho*me["6"] + 3*(me["7"] - rho*me["7"] + 2*me["8"] + 4*rho*me["8"])))/12,((-1 + rho)*me["5"] - 2*(me["6"] + 2*rho*me["6"]))/4,(me["1"] + 2*rho*me["1"] - 2*(2 + rho)*me["2"] + 6*(me["3"] + 2*rho*me["3"] - 2*(2 + rho)*me["4"]))/24,(me["1"] + 2*rho*me["1"] - 2*(2 + rho)*me["2"])/8,(rho**0.5*(me["1"] - 2*(me["2"] - 3*me["3"] + 6*me["4"])))/8,(3*rho**0.5*(me["1"] - 2*me["2"]))/8,-(rho**0.5*(me["1"] + 2*(me["2"] + 3*me["3"] + 6*me["4"])))/2,(-3*rho**0.5*(me["1"] + 2*me["2"]))/2),
        (0,0,0,0,0,0,0,0,0,0,(-3*rho**0.5*(me["5"] - 2*me["6"]))/4,-(rho**0.5*(me["5"] - 2*(me["6"] - 3*me["7"] + 6*me["8"])))/4,((-1 + rho)*me["5"] - 2*(me["6"] + 2*rho*me["6"]))/4,((-1 + rho)*me["5"] - 2*(me["6"] + 2*rho*me["6"] + 3*(me["7"] - rho*me["7"] + 2*me["8"] + 4*rho*me["8"])))/12,(me["1"] + 2*rho*me["1"] - 2*(2 + rho)*me["2"])/8,(me["1"] + 2*rho*me["1"] - 2*(2 + rho)*me["2"] + 6*(me["3"] + 2*rho*me["3"] - 2*(2 + rho)*me["4"]))/24,(3*rho**0.5*(me["1"] - 2*me["2"]))/8,(rho**0.5*(me["1"] - 2*(me["2"] - 3*me["3"] + 6*me["4"])))/8,(-3*rho**0.5*(me["1"] + 2*me["2"]))/2,-(rho**0.5*(me["1"] + 2*(me["2"] + 3*me["3"] + 6*me["4"])))/2),
        (0,0,0,0,0,0,0,0,0,0,(-me["5"] + 2*(me["6"] - 3*me["7"] + 6*me["8"]))/4,(-3*(me["5"] - 2*me["6"]))/4,-(rho**0.5*(me["6"] + 6*me["8"]))/2,(-3*rho**0.5*me["6"])/2,(rho**0.5*(me["1"] - 2*(me["2"] - 3*me["3"] + 6*me["4"])))/8,(3*rho**0.5*(me["1"] - 2*me["2"]))/8,((2 + rho)*me["1"] - 2*(me["2"] + 2*rho*me["2"] - 3*(2 + rho)*me["3"] + 6*(me["4"] + 2*rho*me["4"])))/24,((2 + rho)*me["1"] - 2*(me["2"] + 2*rho*me["2"]))/8,((-4 + rho)*me["1"] - 2*(me["2"] + 2*rho*me["2"] - 3*(-4 + rho)*me["3"] + 6*(me["4"] + 2*rho*me["4"])))/6,((-4 + rho)*me["1"])/2 - (1 + 2*rho)*me["2"]),
        (0,0,0,0,0,0,0,0,0,0,(-3*(me["5"] - 2*me["6"]))/4,(-me["5"] + 2*(me["6"] - 3*me["7"] + 6*me["8"]))/4,(-3*rho**0.5*me["6"])/2,-(rho**0.5*(me["6"] + 6*me["8"]))/2,(3*rho**0.5*(me["1"] - 2*me["2"]))/8,(rho**0.5*(me["1"] - 2*(me["2"] - 3*me["3"] + 6*me["4"])))/8,((2 + rho)*me["1"] - 2*(me["2"] + 2*rho*me["2"]))/8,((2 + rho)*me["1"] - 2*(me["2"] + 2*rho*me["2"] - 3*(2 + rho)*me["3"] + 6*(me["4"] + 2*rho*me["4"])))/24,((-4 + rho)*me["1"])/2 - (1 + 2*rho)*me["2"],((-4 + rho)*me["1"] - 2*(me["2"] + 2*rho*me["2"] - 3*(-4 + rho)*me["3"] + 6*(me["4"] + 2*rho*me["4"])))/6),
        (0,0,0,0,0,0,0,0,0,0,3*(me["5"] - 2*(me["6"] - 3*me["7"] + 6*me["8"])),9*(me["5"] - 2*me["6"]),-2*rho**0.5*(me["5"] - me["6"] + 6*me["7"] - 6*me["8"]),-6*rho**0.5*(me["5"] - me["6"]),-(rho**0.5*(me["1"] + 2*(me["2"] + 3*me["3"] + 6*me["4"])))/2,(-3*rho**0.5*(me["1"] + 2*me["2"]))/2,((-4 + rho)*me["1"] - 2*(me["2"] + 2*rho*me["2"] - 3*(-4 + rho)*me["3"] + 6*(me["4"] + 2*rho*me["4"])))/6,((-4 + rho)*me["1"])/2 - (1 + 2*rho)*me["2"],(2*((14 + rho)*me["1"] - 2*(me["2"] + 2*rho*me["2"] - 3*(14 + rho)*me["3"] + 6*(me["4"] + 2*rho*me["4"]))))/3,2*(14 + rho)*me["1"] - 4*(me["2"] + 2*rho*me["2"])),
        (0,0,0,0,0,0,0,0,0,0,9*(me["5"] - 2*me["6"]),3*(me["5"] - 2*(me["6"] - 3*me["7"] + 6*me["8"])),-6*rho**0.5*(me["5"] - me["6"]),-2*rho**0.5*(me["5"] - me["6"] + 6*me["7"] - 6*me["8"]),(-3*rho**0.5*(me["1"] + 2*me["2"]))/2,-(rho**0.5*(me["1"] + 2*(me["2"] + 3*me["3"] + 6*me["4"])))/2,((-4 + rho)*me["1"])/2 - (1 + 2*rho)*me["2"],((-4 + rho)*me["1"] - 2*(me["2"] + 2*rho*me["2"] - 3*(-4 + rho)*me["3"] + 6*(me["4"] + 2*rho*me["4"])))/6,2*(14 + rho)*me["1"] - 4*(me["2"] + 2*rho*me["2"]),(2*((14 + rho)*me["1"] - 2*(me["2"] + 2*rho*me["2"] - 3*(14 + rho)*me["3"] + 6*(me["4"] + 2*rho*me["4"]))))/3)
    ))

    dflav = {'d': 0, 's': 1, 'b': 2}
    result = 0
    for sector in ("dbcu", "sbcu"):
        CSM, CNP = siegen_basis_wcs(wc_obj, par, sector)
        C = CSM + CNP
        ckm_factor = V[0, dflav[sector[0]]] * V[1,2] # Vud Vcb or Vus Vcb
        prefactor = GF**2 * mb**2 * abs(ckm_factor)**2 * (1 - rho)**2 / (6 * pi)
        # While this expression is manifestly real, we take take the real part to avoid
        # tiny imaginary parts from floating point errors
        result += prefactor * (C @ A_PI_cd @ C.conj() - CSM @ A_PI_cd @ CSM.conj()).real

    return result


def tau_Bp_over_tau_Bd(wc_obj, par):
    r"""Ratio of the B+ over Bd lifetimes based on the SM estimate plus
    the NP contribution from (some) four quark operators."""
    ratio_SM = tau_Bp_over_tau_Bd_SM(par)

    delta_ratio_BSM = gamma_BSM_dim6(wc_obj, par, "B0") - gamma_BSM_dim6(wc_obj, par, "B+")

    return ratio_SM + delta_ratio_BSM * par['tau_B+']


# Observable and Prediction instance
_process_tex = r"B_q \to X"
_process_taxonomy = r'Process :: $b$ hadron decays :: Lifetimes :: $' + _process_tex + r"$"

_obs_name = "tau_B+/tau_Bd"
_obs = flavio.classes.Observable(_obs_name)
_obs.set_description(r"$B^+ / B_d$ lifetime ratio")
_obs.tex = r"$\tau_{B^+} / \tau_{B_d}$"
_obs.add_taxonomy(_process_taxonomy)
flavio.classes.Prediction(_obs_name, tau_Bp_over_tau_Bd)
