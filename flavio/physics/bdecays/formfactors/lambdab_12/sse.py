from math import sqrt
import numpy as np
from flavio.physics.bdecays.formfactors.common import z

def zs(mX, mP, par, q2):
    mLb = par['m_Lambdab']
    mB = par['m_B+']
    zq2 = z(mB, mP, q2, t0=(mLb - mX)**2) # cf. eq. (34) of arXiv:1602.01399
    return np.array([1, zq2, zq2**2])

def pole(ff, mres, q2):
    m = mres[_mresdict[ff]]
    return 1/(1-q2/m**2)

_mresdict = {'fV0': '1-',
             'fVperp': '1-',
             'fT0': '1-',
             'fTperp': '1-',
             'fVt': '0+',
             'fA0': '1+',
             'fAperp': '1+',
             'fT50': '1+',
             'fT5perp': '1+',
             'fAt': '0-',
            }
# resonance masses used in arXiv:1602.01399
_mres = {}
_mres['b->s'] = {'1-': 5.416, '0+': 5.711, '1+': 5.750, '0-': 5.367}

# X: name of the final-state baryon, q: quark-level transition (e.g. 'b->s'),
# P: name of the pseudoscalar meson with the same flavour QN as X (needed for z)
_process_dict = {}
_process_dict['Lambdab->Lambda'] =    {'X': 'Lambda', 'P': 'K+', 'q': 'b->s'}

def ff(process, q2, par, n=2):
    r"""Central value of $\Lambda_b\to X_{1/2}$ form factors in the helicity
    basis and simplified series expansion (SSE) parametrization.

    The helicity basis defines the form factors
    $f_+$, $f_0$, $f_\perp$, $g_+$, $g_0$, $g_\perp$,
    $h_+$, $h_\perp$, $\tilde h_+$, $\tilde h_\perp$,

    The SSE defines
    $$F_i(q^2) = P_i(q^2) \sum_k a_k^i \,z(q^2)^k$$
    where $P_i(q^2)=(1-q^2/m_{R,i}^2)^{-1}$ is a simple pole.
    """
    pd = _process_dict[process]
    mres = _mres[pd['q']]
    mX = par['m_'+pd['X']]
    mP = par['m_'+pd['P']]
    ff = {}
    par_copy = par.copy()
    # implementing the two endpoint relations in (7) and (8) of arXiv:1602.01399
    par_copy[process+' SSE a0_fAperp'] = par_copy[process+' SSE a0_fA0']
    par_copy[process+' SSE a0_fT5perp'] = par_copy[process+' SSE a0_fT50']
    for i in ['fA0', 'fAperp', 'fAt', 'fT0', 'fT50', 'fT5perp', 'fTperp', 'fV0', 'fVperp', 'fVt']:
        a = [ par_copy[process+' SSE ' + 'a' + str(j) + '_' + i] for j in range(n) ]
        ff[i] = pole(i, mres, q2)*np.dot(a, zs(mX, mP, par, q2)[:n])
    return ff
