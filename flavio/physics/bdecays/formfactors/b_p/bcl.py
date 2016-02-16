from math import sqrt
import numpy as np
from flavio.physics.bdecays.formfactors.common import z
from flavio.physics.bdecays.formfactors.b_p.isgurwise import improved_isgur_wise

def pole(ff, mres, q2):
    mresdict = {'f0': 0,'f+': 1,'fT': 1}
    m = mres[mresdict[ff]]
    if m == 0:
        return 1
    return 1/(1-q2/m**2)

mres_lattice = {}
# resonance masses used in 1509.06235
mres_lattice['b->s'] = [5.711, 5.4154];
# resonance masses used in 1505.03925
mres_lattice['b->c'] = [6.420, 6.330];
# resonance masses used in 1507.01618
mres_lattice['b->u'] = [5.319, 5.319]; # this is just mB*

process_dict = {}
process_dict['B->K'] =    {'B': 'B0', 'P': 'K0', 'q': 'b->s'}
process_dict['B->D'] =    {'B': 'B+', 'P': 'D0', 'q': 'b->c'}
process_dict['B->pi'] =   {'B': 'B+', 'P': 'pi0', 'q': 'b->u'}


def param_fplusT(mB, mP, a_i, q2):
    Z = z(mB, mP, q2)
    k = np.arange(len(a_i))
    return ( a_i * (Z**k - (-1)**(k - 3) * k/3 * Z**3) ).sum()

def param_f0(mB, mP, a_i, q2):
    Z = z(mB, mP, q2)
    k = np.arange(len(a_i))
    return ( a_i * Z**k ).sum()

def ff(process, q2, par, n=3):
    r"""Central value of $B\to P$ form factors in the standard convention
    and BCL parametrization (arXiv:0807.2722).

    The standard convention defines the form factors $f_+$, $f_0$, and $f_T$.
    """
    pd = process_dict[process]
    mres = mres_lattice[pd['q']]
    mB = par['m_'+pd['B']]
    mP = par['m_'+pd['P']]
    ff = {}
    a={}
    for i in ['f+', 'fT', 'f0']:
        a[i] = [ par[process + ' BCL' + str(n) + ' a' + str(j) + '_' + i] for j in range(n) ]
    ff['f+'] = pole('f+', mres, q2) * param_fplusT(mB, mP, a['f+'], q2)
    ff['fT'] = pole('fT', mres, q2) * param_fplusT(mB, mP, a['fT'], q2)
    ff['f0'] = pole('f0', mres, q2) * param_f0(mB, mP, a['f0'], q2)
    return ff

def ff_isgurwise(process, q2, par, scale, n=3):
    r"""Central value of $B\to P$ form factors in the standard convention
    and BCL parametrization (arXiv:0807.2722) for $f_0$ and $f_+$, but using
    an improved Isgur-Wise relation in the heavy quark limit for $f_T$.
    """
    pd = process_dict[process]
    mres = mres_lattice[pd['q']]
    mB = par['m_'+pd['B']]
    mP = par['m_'+pd['P']]
    ff = {}
    a={}
    for i in ['f+', 'f0']:
        a[i] = [ par[process + ' BCL' + str(n) + ' a' + str(j) + '_' + i] for j in range(n) ]
    ff['f+'] = pole('f+', mres, q2) * param_fplusT(mB, mP, a['f+'], q2)
    ff['f0'] = pole('f0', mres, q2) * param_f0(mB, mP, a['f0'], q2)
    ff = improved_isgur_wise(q2, ff, par, B=pd['B'], P=pd['P'], scale=scale)
    return ff
