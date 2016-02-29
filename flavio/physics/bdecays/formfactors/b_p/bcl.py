from math import sqrt
import numpy as np
from flavio.physics.bdecays.formfactors.common import z
from flavio.physics.bdecays.formfactors.b_p.isgurwise import improved_isgur_wise

def pole(ff, mres, q2):
    mresdict = {'f0': 0,'f+': 1,'fT': 1}
    m = mres[mresdict[ff]]
    if m == 0 or m is None:
        return 1
    return 1/(1-q2/m**2)

resonance_masses = {}
# resonance masses used in 1509.06235
# this is m_Bs*(0+), m_Bs*(1-)
resonance_masses['B->K'] = [5.711, 5.4154]
# resonance masses used in 1505.03925
resonance_masses['B->D'] = [6.420, 6.330]
# resonance masses used in 1503.07839v2
# this is m_B*(1-)
resonance_masses['B->pi'] = [None, 5.319]
# resonance masses used in 1501.05373v3
# this is m_B*(0+), m_B*(1-)
resonance_masses['Bs->K'] = [5.63, 5.3252]

# the following dict maps transitions to mesons. Note that it doesn't really
# matter whether the charged or neutral B/K/pi are used here. We don't
# distinguish between charged and neutral form factors anyway.
process_dict = {}
process_dict['B->K'] =    {'B': 'B0', 'P': 'K0',}
process_dict['Bs->K'] =    {'B': 'Bs', 'P': 'K+',}
process_dict['B->D'] =    {'B': 'B+', 'P': 'D0',}
process_dict['B->pi'] =   {'B': 'B+', 'P': 'pi0',}


def param_fplusT(mB, mP, a_i, q2, t0=None):
    Z = z(mB, mP, q2, t0)
    n = len(a_i)
    k = np.arange(n)
    return ( a_i * (Z**k - (-1)**(k - n) * k/n * Z**n) ).sum()

def param_f0(mB, mP, a_i, q2, t0=None):
    Z = z(mB, mP, q2, t0)
    k = np.arange(len(a_i))
    return ( a_i * Z**k ).sum()

def ff(process, q2, par, n=3, t0=None):
    r"""Central value of $B\to P$ form factors in the standard convention
    and BCL parametrization (arXiv:0807.2722).

    The standard convention defines the form factors $f_+$, $f_0$, and $f_T$.
    """
    pd = process_dict[process]
    mres = resonance_masses[process]
    mB = par['m_'+pd['B']]
    mP = par['m_'+pd['P']]
    ff = {}
    a={}
    for i in ['f+', 'fT', 'f0']:
        a[i] = [ par[process + ' BCL' + ' a' + str(j) + '_' + i] for j in range(n) ]
    ff['f+'] = pole('f+', mres, q2) * param_fplusT(mB, mP, a['f+'], q2, t0)
    ff['fT'] = pole('fT', mres, q2) * param_fplusT(mB, mP, a['fT'], q2, t0)
    ff['f0'] = pole('f0', mres, q2) * param_f0(mB, mP, a['f0'], q2, t0)
    return ff

def ff_isgurwise(process, q2, par, scale, n=3, t0=None):
    r"""Central value of $B\to P$ form factors in the standard convention
    and BCL parametrization (arXiv:0807.2722) for $f_0$ and $f_+$, but using
    an improved Isgur-Wise relation in the heavy quark limit for $f_T$.
    """
    pd = process_dict[process]
    mres = resonance_masses[process]
    mB = par['m_'+pd['B']]
    mP = par['m_'+pd['P']]
    ff = {}
    a={}
    for i in ['f+', 'f0']:
        a[i] = [ par[process + ' BCL' + ' a' + str(j) + '_' + i] for j in range(n) ]
    ff['f+'] = pole('f+', mres, q2) * param_fplusT(mB, mP, a['f+'], q2, t0)
    ff['f0'] = pole('f0', mres, q2) * param_f0(mB, mP, a['f0'], q2, t0)
    ff = improved_isgur_wise(process, q2, ff, par, scale=scale)
    return ff
