r"""Functions for Higgs production.

The numerical coefficients have been obtained with MadGraph_aMC@NLO v2.6.5
along with SMEFTsim v2 in the alpha scheme.
"""


def ggF(C):
    r"""Higgs production from gluon fusion normalized to the SM"""
    np = (+35.86 * C['phiG']
          +0.121 * (C['phiBox'] - C['phiD'] / 4.)
          +0.061 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          +0.035 * C['uphi_33']
          -0.034 * C['uphi_22']
          -0.017 * C['dphi_33']
          -0.043 * C['ephi_33']
          -0.008 * C['ephi_22']
          )
    return 1 + 1e6 * np.real

def hw(C):
    r"""Higgs production associated with a $W$ normalized to the SM"""
    np = (+0.745 * C['phiW']
          -0.241 * C['phiWB']
          -0.135 * C['phiD']
          +0.101 * C['phiBox']
          +0.168 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
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
    np = (-0 * C['phiB']
          -0.005 * C['phiW']
          -0.021 * C['phiWB']
          -0.011 * C['phiD']
          +0.008 * C['phiBox']
          +0.018 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          )
    return 1 + 1e6 * np.real
