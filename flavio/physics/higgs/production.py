r"""Functions for Higgs production.

Most of the numerical coefficients have been obtained with MadGraph_aMC@NLO v2.6.5
along with SMEFTsim v2 in the alpha scheme.
"""

import flavio

def ggF(C):
    r"""Higgs production from gluon fusion normalized to the SM"""
    # obtained from an analytical one-loop calculation
    flavio.citations.register("Falkowski:2019hvp")
    np = (+35.86 * C['phiG']
          +0.121 * (C['phiBox'] - C['phiD'] / 4.)
          +0.061 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          -0.129 * C['uphi_33']
          +0.123 * C['uphi_22']
          +0.239 * C['dphi_33']
          +0.025 * C['dphi_22']
          )
    return 1 + 1e6 * np.real

def hw(C):
    r"""Higgs production associated with a $W$ normalized to the SM"""
    flavio.citations.register("Falkowski:2019hvp")
    np = (+0.891 * C['phiW']
          -0.187 * C['phiWB']
          -0.115 * C['phiD']
          +0.121 * C['phiBox']
          +0.173 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          +1.85 * C['phiq3_11']
          +0.126 * C['phiq3_22']
          )
    return 1 + 1e6 * np.real

def hz(C):
    r"""Higgs production associated with a $Z$ normalized to the SM"""
    flavio.citations.register("Falkowski:2019hvp")
    np = (+0.098 * C['phiB']
          +0.721 * C['phiW']
          +0.217 * C['phiWB']
          -0.015 * C['phiD']
          +0.122 * C['phiBox']
          +0.152 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          -0.187 * C['phiq1_11']
          +1.699 * C['phiq3_11']
          +0.456 * C['phiu_11']
          -0.148 * C['phid_11']
          +0.044 * C['phiq1_22']
          +0.16 * C['phiq3_22']
          +0.028 * C['phiu_22']
          -0.02 * C['phid_22']
          )
    return 1 + 1e6 * np.real

def hv(C):
    r"""Higgs production associated with a $W$ or $Z$ normalized to the SM"""
    # Wh xsec at 14 TeV in pb, https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt1314TeV2014#s_13_0_TeV
    xw_sm = 1.380
    # Zh xsec at 14 TeV in pb, https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt1314TeV2014#s_13_0_TeV
    xz_sm = 0.8696
    d_hw = hw(C) - 1
    d_hz = hz(C) - 1
    return (xw_sm * (1 + d_hw) + xz_sm * (1 + d_hz)) / (xw_sm + xz_sm)

def tth(C):
    r"""Higgs production associated with a top pair normalized to the SM"""
    flavio.citations.register("Falkowski:2019hvp")
    np = (-0.030 * C['phiD']
          +0.118 * C['phiBox']
          -0.853 * C['uG_33']
          +0.146 * C['G']
          +0.557 * C['phiG']
          +0.060 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          -0.119 * C['uphi_33'])
    return 1 + 1e6 * np.real

def vv_h(C):
    r"""Higgs production from vector boson fusion normalized to the SM"""
    flavio.citations.register("Falkowski:2019hvp")
    np = (-0.002 * C['phiB']
          -0.088 * C['phiW']
          -0.319 * C['phiWB']
          -0.168 * C['phiD']
          +0.121 * C['phiBox']
          +0.277 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          +0.014 * C['phiq1_11']
          -0.384 * C['phiq3_11']
          -0.027 * C['phiu_11']
          +0.008 * C['phid_11']
          -0.004 * C['phiq1_22']
          -0.075 * C['phiq3_22']
          -0.004 * C['phiu_22']
          +0.002 * C['phid_22']
          )
    return 1 + 1e6 * np.real
