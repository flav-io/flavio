from math import sqrt, pi
from flavio.physics.bdecays.common import meson_ff
from flavio.physics.bdecays.formfactors import hqet
from flavio.physics.bdecays.formfactors import common
from flavio.classes import AuxiliaryQuantity
from flavio.physics.running import running

process_dict = {}
process_dict['B->D*'] = {'B': 'B0', 'V': 'D*+', 'q': 'b->c'}


def h_to_A(mB, mV, h, q2):
    """Convert HQET form factors to the standard basis.

    See e.g. arXiv:1309.0301, eqs. (38), (39)"""
    ff = {}
    pre = 1 / 2 / sqrt(mB * mV)
    ff['V'] = pre * (mB + mV) * h['V']
    ff['A1'] = pre * ((mB + mV)**2 - q2) / (mB + mV) * h['A1']
    ff['A2'] = pre * (mB + mV) * (h['A3'] + mV / mB * h['A2'])
    ff['A0'] = pre * (((mB + mV)**2 - q2) / (2 * mV) * h['A1']
                      - (mB**2 - mV**2 + q2) / (2 * mB) * h['A2']
                      - (mB**2 - mV**2 - q2) / (2 * mV) * h['A3'])
    ff['T1'] = pre * ((mB + mV) * h['T1'] - (mB - mV) * h['T2'])
    ff['T2'] = pre * (((mB + mV)**2 - q2) / (mB + mV) * h['T1']
                      - ((mB - mV)**2 - q2) / (mB - mV) * h['T2'])
    ff['T3'] = pre * ((mB - mV) * h['T1'] - (mB + mV) * h['T2']
                      - 2 * (mB**2 - mV**2) / mB * h['T3'])
    # conversion from A_1, A_2 to A_12
    ff['A12'] = ((ff['A1'] * (mB + mV)**2 * (mB**2 - mV**2 - q2)
                 - ff['A2'] * (mB**4 + (mV**2 - q2)**2
                 - 2 * mB**2 * (mV**2 + q2)))
                 / (16. * mB * mV**2 * (mB + mV)))
    del ff['A2']
    # conversion from T_2, T_3 to T_23
    ff['T23'] = ((mB**2 - mV**2) * (mB**2 + 3 * mV**2 - q2) * ff['T2']
                 - (mB**4 + (mV**2 - q2)**2
                 - 2 * mB**2 * (mV**2 + q2)) * ff['T3']
                 ) / (8 * mB * (mB - mV) * mV**2)
    del ff['T3']
    return ff

def ff(process, q2, par, scale):
    r"""Central value of $B\to V$ form factors in the lattice convention
    CLN parametrization.

    See arXiv:1703.05330.
    """
    pd = process_dict[process]
    mB = par['m_' + pd['B']]
    mV = par['m_' + pd['V']]
    w = max((mB**2 + mV**2 - q2) / (2 * mB * mV), 1)
    phqet = hqet.get_hqet_parameters(par, scale)
    ash = phqet['ash']
    epsc = phqet['epsc']
    epsb = phqet['epsb']
    zc = phqet['zc']
    # eq. (22) of arXiv:0809.0222
    CV1 = hqet.CV1(w, zc)
    CV2 = hqet.CV2(w, zc)
    CV3 = hqet.CV3(w, zc)
    CA1 = hqet.CA1(w, zc)
    CA2 = hqet.CA2(w, zc)
    CA3 = hqet.CA3(w, zc)
    CT1 = hqet.CT1(w, zc)
    CT2 = hqet.CT2(w, zc)
    CT3 = hqet.CT3(w, zc)
    L = hqet.L(par, w)
    # leading, universal Isgur-Wise function
    rho2 = par['CLN rho2_xi']
    c = par['CLN c_xi']
    z = common.z(mB, mV, q2, t0='tm')
    xi = 1 - 8 * rho2 * z + (64 * c - 16 * rho2) * z**2
    h = {}
    h['V'] = xi * (1 + ash * CV1
                   + epsc * (L[2] - L[5])
                   + epsb * (L[1] - L[4]))
    h['A1'] = xi * (1 + ash * CA1
                    + epsc * (L[2] - L[5] * (w - 1)/(w + 1))
                    + epsb * (L[1] - L[4] * (w - 1)/(w + 1))
                    + epsc**2 * par['B->D* CLN deltac_hA1'])
    h['A2'] = xi * (ash * CA2 + epsc * (L[3] + L[6]))
    h['A3'] = xi * (1 + ash * (CA1 + CA3)
                    + epsc * (L[2] - L[3] + L[6] - L[5])
                    + epsb * (L[1] - L[4]))
    h['T1'] = xi * (1 + ash * (CT1 + (w - 1)/2 * (CT2 - CT3))
                    + epsc * L[2]
                    + epsb * L[1]
                    + epsc**2 * par['B->D* CLN deltac_hT1'])
    h['T2'] = xi * (ash * (w + 1)/2 * (CT2 + CT3)
                    + epsc * L[5]
                    - epsb * L[4])
    h['T3'] = xi * (ash * CT2
                    + epsc * (L[6] - L[3]))
    return h_to_A(mB, mV, h, q2)
