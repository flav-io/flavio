r"""Functions for Higgs production.

The numerical coefficients have been obtained with MadGraph_aMC@NLO v2.6.5
along with SMEFTsim v2 in the alpha scheme.
"""


def ggF(C):
    r"""Higgs production from gluon fusion normalized to the SM"""
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
    np = (+0.891 * C['phiW']
          -0.187 * C['phiWB']
          -0.115 * C['phiD']
          +0.121 * C['phiBox']
          +0.173 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          )
    return 1 + 1e6 * np.real

def hz(C):
    r"""Higgs production associated with a $Z$ normalized to the SM"""
    np = (+0.098 * C['phiB']
          +0.721 * C['phiW']
          +0.217 * C['phiWB']
          -0.015 * C['phiD']
          +0.122 * C['phiBox']
          +0.152 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          )
    return 1 + 1e6 * np.real

def tth(C):
    r"""Higgs production associated with a top pair normalized to the SM"""
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
    np = (-0.002 * C['phiB']
          -0.088 * C['phiW']
          -0.319 * C['phiWB']
          -0.168 * C['phiD']
          +0.121 * C['phiBox']
          +0.277 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          )
    return 1 + 1e6 * np.real
