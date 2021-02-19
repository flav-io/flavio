import numpy as np
from functools import lru_cache
import flavio
from flavio.physics.bdecays.formfactors.common import z
from flavio.config import config


def pole(ff, mres, q2):
    if mres == 0 or mres is None:
        return 1
    return 1/(1-q2/mres**2)


# the following dict maps transitions to mesons. Note that it doesn't really
# matter whether the charged or neutral B/K/pi are used here. We don't
# distinguish between charged and neutral form factors anyway.
process_dict = {}
process_dict['B->K'] = {'B': 'B0', 'P': 'K0'}
process_dict['Bs->K'] = {'B': 'Bs', 'P': 'K+'}
process_dict['B->D'] = {'B': 'B+', 'P': 'D0'}
process_dict['B->pi'] = {'B': 'B+', 'P': 'pi0'}


@lru_cache(maxsize=config['settings']['cache size'])
def zs(mB, mP, q2, t0):
    zq2 = z(mB, mP, q2, t0)
    z0 = z(mB, mP, 0, t0)
    return np.array([1, zq2-z0, (zq2-z0)**2])


def ff(process, q2, par, n=3, t0=None):
    r"""Central value of $B\to P$ form factors in the standard convention
    and BSZ parametrization (arXiv:1811.00983).

    The standard convention defines the form factors $f_+$, $f_0$, and $f_T$.
    """
    flavio.citations.register("Gubernari:2018wyi")
    pd = process_dict[process]
    mpl = par[process + ' BCL m+']
    m0 = par[process + ' BCL m0']
    mB = par['m_' + pd['B']]
    mP = par['m_' + pd['P']]
    a = {}
    ff = {}
    for i in ['f+', 'fT']:
        a[i] = [par[process + ' BSZ a' + str(j) + '_' + i] for j in range(n)]
    # for f0, only a1,... are taken from the parameters,
    # a0 is chosen to fulfill the kinematic constraint f+(0)=f0(0)
    a0_f0 = par[process + ' BSZ a0_f+']
    a['f0'] = [a0_f0] + [par[process + ' BSZ a' + str(j) + '_f0'] for j in range(1, n)]
    # evaluate FFs
    ff['f+'] = pole('f+', mpl, q2) * np.dot(a['f+'], zs(mB, mP, q2, t0=t0)[:n])
    ff['fT'] = pole('fT', mpl, q2) * np.dot(a['fT'], zs(mB, mP, q2, t0=t0)[:n])
    ff['f0'] = pole('f0', m0, q2) * np.dot(a['f0'], zs(mB, mP, q2, t0=t0)[:n])
    return ff
