r"""BSZ parametrization of $D\to \pi,K$ form factors.

Taken from `flavio.physics.bdecays.formfactors.b_p.bsz.py`
"""

import numpy as np
import flavio
from flavio.physics.bdecays.formfactors.b_p.bsz import zs, pole
from flavio.physics.ddecays.formfactors.bcl import process_dict


def ff(process, q2, par, n=3, t0=None):
    r"""Central value of $D\to P$ form factors in the standard convention
    and BSZ parametrization (arXiv:1811.00983).

    The standard convention defines the form factors $f_+$, $f_0$, and $f_T$.
    """
    flavio.citations.register("Gubernari:2018wyi")
    pd = process_dict[process]
    mpl = par[process + ' BCL m+']
    m0 = par[process + ' BCL m0']
    mB = par['m_' + pd['D']]
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
