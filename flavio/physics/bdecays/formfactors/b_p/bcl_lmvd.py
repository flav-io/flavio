import numpy as np
import flavio
from flavio.physics.bdecays.formfactors.common import z

def pole(ff, mres, q2):
    if mres == 0 or mres is None:
        return 1
    return 1/(1-q2/mres**2)

# the following dict maps transitions to mesons. Note that it doesn't really
# matter whether the charged or neutral B/K/pi are used here. We don't
# distinguish between charged and neutral form factors anyway.
process_dict = {}
process_dict['B->pi'] =   {'B': 'B+', 'P': 'pi0'}


def param_fplusT(mB, mP, b_i, q2, t0=None):
    # b_i[0] is f(q^2 = 0), b_i[1:] should be multiplied with b_i[0]
    Z = z(mB, mP, q2, t0)
    Z0 = z(mB, mP, 0, t0)
    n = len(b_i)
    k = np.arange(1, n, 1)
    return b_i[0]*(1+( b_i[1:] * (Z**k-Z0**k - (-1.)**(k - n) * k/n * (Z**n-Z0**n)) ).sum())

def param_f0(mB, mP, b_i, q2, t0=None):
    # b_i[0] is f(q^2 = 0), b_i[1:] should be multiplied with b_i[0]
    Z = z(mB, mP, q2, t0)
    Z0 = z(mB, mP, 0, t0)
    n = len(b_i)
    k = np.arange(1 ,n, 1)
    return b_i[0]*(1+( b_i[1:] * (Z**k -Z0**k)).sum())

def ff(process, q2, par, n=4, t0=None):
    r"""Central value of $B\to P$ form factors in the standard convention
    and BCL parametrization from arXiv:2102.07233.

    The standard convention defines the form factors $f_+$, $f_0$, and $f_T$.
    """
    flavio.citations.register("Leljak:2021vte")
    pd = process_dict[process]
    mpl = par[process + ' BCL LMVD m+']
    m0 = par[process + ' BCL LMVD m0']
    mB = par[process + ' BCL LMVD m_' + pd['B']]
    mP = par[process + ' BCL LMVD m_' + pd['P']]
    ff = {}
    b={}
    for i in ['f+', 'fT']:
        b[i] = [par[process + ' BCL' + ' f' + f'_{i[1:]}(0)']]+ [ par[process + ' BCL' + ' b' f'_{i[1:]}^{j}'] for j in range(1, n)]
    # evaluate FFs
    ff['f+'] = pole('f+', mpl, q2) * param_fplusT(mB, mP, b['f+'], q2, t0)
    ff['fT'] = pole('fT', mpl, q2) * param_fplusT(mB, mP, b['fT'], q2, t0)

    # f0 is modified
    b['f0'] = [par[process + ' BCL' + ' f' + f'_+(0)']]+ [ par[process + ' BCL' + ' b' + f'_0^{j}'] for j in range(1, n+1)] # note the +1
    ff['f0'] = pole('f0', m0, q2) * param_f0(mB, mP, b['f0'], q2, t0)
    return ff
