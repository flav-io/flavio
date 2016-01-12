from math import sqrt
import numpy as np
from flavio.physics.bdecays.formfactors.common import z

def pole(ff, mres, q2):
    mresdict = {'f0': 0,'f+': 1,'fT': 1}
    m = mres[mresdict[ff]]
    if m == 0:
        return 1
    return 1/(1-q2/m**2)

# resonance masses used in 1509.06235
mres_lattice = {}
mres_lattice['b->s'] = [5.711, 5.4154];

process_dict = {}
process_dict['B->K'] =    {'B': 'Bd', 'P': 'K0', 'q': 'b->s'}


def param_fplusT(mB, mP, a_i, q2):
    Z = z(mB, mP, q2)
    k = np.arange(len(a_i))
    return ( a_i * (Z**k - (-1)**(k - 3) * k/3 * Z**3) ).sum()

def param_f0(mB, mP, a_i, q2):
    Z = z(mB, mP, q2)
    k = np.arange(len(a_i))
    return ( a_i * Z**k ).sum()

def ff(process, q2, par):
    r"""Central value of $B\to P$ form factors in the standard convention
    and lattice parametrization.

    The standard convention defines the form factors $f_+$, $f_0$, and $f_T$.

    The lattice parametrization ...

    Parameters
    ----------
    process_dict : dict
        Dictionary of the form {'mres': dict, 'mB': float, 'mV': float}
        containing the initial and final state meson masses as well as the
        resonance mass to be used in the pole.
    a_i : array-like
        a two- or three-dimensional vector containing the series expansion
        coefficients.
    q2 : float
        momentum transfer squared $q^2$
    """
    pd = process_dict[process]
    mres = mres_lattice[pd['q']]
    mB = par[('mass',pd['B'])]
    mP = par[('mass',pd['P'])]
    ff = {}
    a={}
    for i in ['f+', 'fT', 'f0']:
        a[i] = [ par[('formfactor', 'B->K', 'a' + str(j) + '_' + i)] for j in range(2) ]
    ff['f+'] = pole('f+', mres, q2) * param_fplusT(mB, mP, a['f+'], q2)
    ff['fT'] = pole('fT', mres, q2) * param_fplusT(mB, mP, a['fT'], q2)
    ff['f0'] = pole('f0', mres, q2) * param_f0(mB, mP, a['f0'], q2)
    return ff
