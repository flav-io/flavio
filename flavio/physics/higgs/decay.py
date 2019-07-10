r"""Functions for Higgs decay.

The numerical coefficients have been obtained with MadGraph_aMC@NLO v2.6.5
along with SMEFTsim v2 in the alpha scheme.
"""


def h_gg(C):
    r"""Higgs decay to two gluons normalized to the SM"""
    np = (+39.29 * C['phiG']
          +0.121 * (C['phiBox'] - C['phiD'] / 4.)
          +0.061 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          +0.035 * C['uphi_33']
          -0.034 * C['uphi_22']
          -0.017 * C['dphi_33']
          -0.043 * C['ephi_33']
          -0.008 * C['ephi_22']
          )
    return 1 + 1e6 * np.real

def h_gaga(C):
    r"""Higgs decay to two photons normalized to the SM"""
    np = (-46.4 * C['phiB']
          -14.14 * C['phiW']
          +25.62 * C['phiWB']
          +0.121 * (C['phiBox'] - C['phiD'] / 4.)
          +0.061 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          -0.129 * C['uphi_33']
          +0.123 * C['uphi_22']
          +0.239 * C['dphi_33']
          +0.025 * C['dphi_22']
          )
    return 1 + 1e6 * np.real

def h_ww(C):
    r"""Higgs decay to two $W$ bosons normalized to the SM"""
    np = (-0.028 * C['phiW']
          -0.119 * C['phiWB']
          -0.063 * C['phiD']
          +0.037 * C['phiBox']
          +0.088 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          )
    return 1 + 1e6 * np.real

def h_zz(C):
    r"""Higgs decay to $Z$ bosons normalized to the SM"""
    np = (+0.329 * C['phiB']
          -0.386 * C['phiW']
          +0.149 * C['phiWB']
          -0.039 * C['phiD']
          +0.118 * C['phiBox']
          +0.198 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          )
    return 1 + 1e6 * np.real

def h_zga(C):
    r"""Higgs decay to $Z\gamma$ normalized to the SM"""
    np = (+14.89 * C['phiB']
          -14.89 * C['phiW']
          +9.377 * C['phiWB']
          )
    return 1 + 1e6 * np.real

def h_bb(C):
    r"""Higgs decay to two $b$ quarks normalized to the SM"""
    np = (-0.03 * C['phiD']
          +0.121 * C['phiBox']
          +0.061 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          -5.05 * C['dphi_33'])
    return 1 + 1e6 * np.real

def h_tautau(C):
    r"""Higgs decay to two taus normalized to the SM"""
    np = (-0.03 * C['phiD']
          +0.121 * C['phiBox']
          +0.061 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          -11.88 * C['ephi_33'])
    return 1 + 1e6 * np.real

def h_mumu(C):
    r"""Higgs decay to two muons normalized to the SM"""
    np = (-0.03 * C['phiD']
          +0.121 * C['phiBox']
          +0.061 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          -199.8 * C['ephi_22'])
    return 1 + 1e6 * np.real
