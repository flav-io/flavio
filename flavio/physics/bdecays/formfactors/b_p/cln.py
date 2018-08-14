from flavio.physics.bdecays.formfactors import hqet
from flavio.physics.bdecays.formfactors import common
from math import sqrt

process_dict = {}
process_dict['B->D'] = {'B': 'B0', 'q': 'b->c', 'P': 'D+'}


def h_to_f(mB, mP, h, q2):
    """Convert HQET form factors to the standard basis.

    See e.g. arXiv:1309.0301, eq. (31)"""
    ff = {}
    r = mP / mB
    ff['f+'] = ((r + 1) * h['h+'] + (r - 1) * h['h-']) / (2 * sqrt(r))
    fminus = ((r - 1) * h['h+'] + (r + 1) * h['h-']) / (2 * sqrt(r))
    ff['f0'] = ff['f+'] + fminus * q2 / (mB**2 - mP**2)
    ff['fT'] = (r + 1) / (2 * sqrt(r)) * h['hT']
    return ff


def ff(process, q2, par, scale):
    r"""Central value of $B\to P$ form factors in the lattice convention
    CLN parametrization.

    See arXiv:hep-ph/9712417 and arXiv:1703.05330.
    """
    pd = process_dict[process]
    mB = par['m_' + pd['B']]
    mP = par['m_' + pd['P']]
    w = max((mB**2 + mP**2 - q2) / (2 * mB * mP), 1)
    phqet = hqet.get_hqet_parameters(par, scale)
    ash = phqet['ash']
    epsc = phqet['epsc']
    epsb = phqet['epsb']
    zc = phqet['zc']
    CV1 = hqet.CV1(w, zc)
    CV2 = hqet.CV2(w, zc)
    CV3 = hqet.CV3(w, zc)
    CT1 = hqet.CT1(w, zc)
    CT2 = hqet.CT2(w, zc)
    CT3 = hqet.CT3(w, zc)
    L = hqet.L(par, w)
    # leading, universal Isgur-Wise function
    rho2 = par['CLN rho2_xi']
    c = par['CLN c_xi']
    z = common.z(mB, mP, q2, t0='tm')
    xi = 1 - 8 * rho2 * z + (64 * c - 16 * rho2) * z**2
    h = {}
    h['h+'] = xi * (1 + ash * (CV1 + (w + 1) / 2 * (CV2 + CV3))
                    + (epsc + epsb) * L[1]
                    + epsc**2 * par['B->D CLN deltac_h+'])
    h['h-'] = xi * (ash * (w + 1) / 2 * (CV2 - CV3)
                    + (epsc - epsb) * L[4])
    h['hT'] = xi * (1 + ash * (CT1 - CT2 + CT3)
                    + (epsc + epsb) * (L[1] - L[4]))
    return h_to_f(mB, mP, h, q2)
