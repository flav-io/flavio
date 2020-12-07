r"""Functions for Higgs decay.

Most of the numerical coefficients have been obtained with MadGraph_aMC@NLO v2.6.5
along with SMEFTsim v2 in the alpha scheme.
"""

import flavio


def h_gg(C):
    r"""Higgs decay to two gluons normalized to the SM"""
    # obtained from an analytical one-loop calculation
    flavio.citations.register("Falkowski:2019hvp")
    np = (+39.29 * C['phiG']
          +0.121 * (C['phiBox'] - C['phiD'] / 4.)
          +0.061 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          -0.129 * C['uphi_33']
          +0.123 * C['uphi_22']
          +0.239 * C['dphi_33']
          +0.025 * C['dphi_22']
          )
    return 1 + 1e6 * np.real

def h_gaga(C):
    r"""Higgs decay to two photons normalized to the SM"""
    # obtained from an analytical one-loop calculation
    flavio.citations.register("Falkowski:2019hvp")
    np = (-45.78 * C['phiB']
          -13.75 * C['phiW']
          +(25.09 - 0.242) * C['phiWB']  # tree - loop
          +0.121 * C['phiBox']
          -0.141 * C['phiD']
          +0.127 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          +0.035 * C['uphi_33']
          -0.034 * C['uphi_22']
          -0.017 * C['dphi_33']
          -0.043 * C['ephi_33']
          -0.008 * C['ephi_22']
          )
    return 1 + 1e6 * np.real

def h_ww(C):
    r"""Higgs decay to two $W$ bosons normalized to the SM"""
    flavio.citations.register("Falkowski:2019hvp")
    np = (-0.092 * C['phiW']
          -0.386 * C['phiWB']
          -0.205 * C['phiD']
          +0.121 * C['phiBox']
          +0.144 * C['ll_1221']
          -0.167 * C['phil3_11']
          -0.167 * C['phil3_22']
          )
    return 1 + 1e6 * np.real

def h_zz(C):
    r"""Higgs decay to $Z$ bosons normalized to the SM"""
    flavio.citations.register("Falkowski:2019hvp")
    np = (+0.329 * C['phiB']
          -0.386 * C['phiW']
          +0.149 * C['phiWB']
          -0.039 * C['phiD']
          +0.118 * C['phiBox']
          +0.198 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          )
    return 1 + 1e6 * np.real

def h_vv(C):
    r"""Higgs decay to $W$ or $Z$ bosons normalized to the SM"""
    br_ww = flavio.physics.higgs.width.BR_SM['WW']
    br_zz = flavio.physics.higgs.width.BR_SM['ZZ']
    d_ww = h_ww(C) - 1
    d_zz = h_zz(C) - 1
    return (br_ww * (1 + d_ww) + br_zz * (1 + d_zz)) / (br_ww + br_zz)

def h_zga(C):
    r"""Higgs decay to $Z\gamma$ normalized to the SM"""
    flavio.citations.register("Falkowski:2019hvp")
    np = (+14.89 * C['phiB']
          -14.89 * C['phiW']
          +9.377 * C['phiWB']
          )
    return 1 + 1e6 * np.real

def h_bb(C):
    r"""Higgs decay to two $b$ quarks normalized to the SM"""
    flavio.citations.register("Falkowski:2019hvp")
    np = (-0.03 * C['phiD']
          +0.121 * C['phiBox']
          +0.061 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          -5.05 * C['dphi_33'])
    return 1 + 1e6 * np.real

def h_cc(C):
    r"""Higgs decay to two charm quarks normalized to the SM"""
    flavio.citations.register("Falkowski:2019hvp")
    np = (-0.03 * C['phiD']
          +0.121 * C['phiBox']
          +0.061 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          -16.49 * C['uphi_22'])
    return 1 + 1e6 * np.real

def h_tautau(C):
    r"""Higgs decay to two taus normalized to the SM"""
    flavio.citations.register("Falkowski:2019hvp")
    np = (-0.03 * C['phiD']
          +0.121 * C['phiBox']
          +0.061 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          -11.88 * C['ephi_33'])
    return 1 + 1e6 * np.real

def h_mumu(C):
    r"""Higgs decay to two muons normalized to the SM"""
    flavio.citations.register("Falkowski:2019hvp")
    np = (-0.03 * C['phiD']
          +0.121 * C['phiBox']
          +0.061 * (C['ll_1221'] / 2 - C['phil3_22'] - C['phil3_11'])
          -199.8 * C['ephi_22'])
    return 1 + 1e6 * np.real
