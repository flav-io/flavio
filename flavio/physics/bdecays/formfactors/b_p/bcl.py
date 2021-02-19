from math import sqrt
import numpy as np
import flavio
from flavio.physics.bdecays.formfactors.common import z
from flavio.physics.bdecays.formfactors.b_p.isgurwise import isgur_wise

def pole(ff, mres, q2):
    if mres == 0 or mres is None:
        return 1
    return 1/(1-q2/mres**2)

# the following dict maps transitions to mesons. Note that it doesn't really
# matter whether the charged or neutral B/K/pi are used here. We don't
# distinguish between charged and neutral form factors anyway.
process_dict = {}
process_dict['B->K'] =    {'B': 'B0', 'P': 'K0'}
process_dict['Bs->K'] =    {'B': 'Bs', 'P': 'K+'}
process_dict['B->D'] =    {'B': 'B+', 'P': 'D0'}
process_dict['B->pi'] =   {'B': 'B+', 'P': 'pi0'}


def param_fplusT(mB, mP, a_i, q2, t0=None):
    Z = z(mB, mP, q2, t0)
    n = len(a_i)
    k = np.arange(n)
    return ( a_i * (Z**k - (-1.)**(k - n) * k/n * Z**n) ).sum()

def param_f0(mB, mP, a_i, q2, t0=None):
    Z = z(mB, mP, q2, t0)
    k = np.arange(len(a_i))
    return ( a_i * Z**k ).sum()

def ff(process, q2, par, n=3, t0=None):
    r"""Central value of $B\to P$ form factors in the standard convention
    and BCL parametrization (arXiv:0807.2722).

    The standard convention defines the form factors $f_+$, $f_0$, and $f_T$.
    """
    flavio.citations.register("Bourrely:2008za")
    pd = process_dict[process]
    mpl = par[process + ' BCL m+']
    m0 = par[process + ' BCL m0']
    mB = par['m_'+pd['B']]
    mP = par['m_'+pd['P']]
    ff = {}
    a={}
    for i in ['f+', 'fT']:
        a[i] = [ par[process + ' BCL' + ' a' + str(j) + '_' + i] for j in range(n) ]
    # only the first n-1 parameters for f0 are taken from par
    # the nth one is chosen to fulfill the kinematic constraint f+(0)=f0(0)
    a['f0'] = [ par[process + ' BCL' + ' a' + str(j) + '_f0'] for j in range(n-1) ]
    fplus_q20 = pole('f+', mpl, 0) * param_fplusT(mB, mP, a['f+'], 0, t0)
    f0_q20 = pole('f0', m0, 0) * param_f0(mB, mP, a['f0'], 0, t0)
    an_f0 = (f0_q20-fplus_q20)/z(mB, mP, 0, t0)**(n-1)
    a['f0'].append(an_f0)
    # evaluate FFs
    ff['f+'] = pole('f+', mpl, q2) * param_fplusT(mB, mP, a['f+'], q2, t0)
    ff['fT'] = pole('fT', mpl, q2) * param_fplusT(mB, mP, a['fT'], q2, t0)
    ff['f0'] = pole('f0', m0, q2) * param_f0(mB, mP, a['f0'], q2, t0)
    return ff

def ff_isgurwise(process, q2, par, scale, n=3, t0=None):
    r"""Central value of $B\to P$ form factors in the standard convention
    and BCL parametrization (arXiv:0807.2722) for $f_0$ and $f_+$, but using
    an improved Isgur-Wise relation in the heavy quark limit for $f_T$.
    """
    flavio.citations.register("Bourrely:2008za")
    pd = process_dict[process]
    mpl = par[process + ' BCL m+']
    m0 = par[process + ' BCL m0']
    mB = par['m_'+pd['B']]
    mP = par['m_'+pd['P']]
    ff = {}
    a={}
    a['f+'] = [ par[process + ' BCL' + ' a' + str(j) + '_f+'] for j in range(n) ]
    # only the first n-1 parameters for f0 are taken from par
    # the nth one is chosen to fulfill the kinematic constraint f+(0)=f0(0)
    a['f0'] = [ par[process + ' BCL' + ' a' + str(j) + '_f0'] for j in range(n-1) ]
    fplus_q20 = pole('f+', mpl, 0) * param_fplusT(mB, mP, a['f+'], 0, t0)
    f0_q20 = pole('f0', m0, 0) * param_f0(mB, mP, a['f0'], 0, t0)
    an_f0 = (fplus_q20-f0_q20)/z(mB, mP, 0, t0)**(n-1)
    a['f0'].append(an_f0)
    # evaluate FFs
    ff['f+'] = pole('f+', mpl, q2) * param_fplusT(mB, mP, a['f+'], q2, t0)
    ff['f0'] = pole('f0', m0, q2) * param_f0(mB, mP, a['f0'], q2, t0)
    ff = isgur_wise(process, q2, ff, par, scale=scale)
    return ff
