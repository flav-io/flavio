from math import sqrt
import numpy as np
from flavio.physics.bdecays.formfactors.common import z

def zs(mB, mV, q2, t0):
    zq2 = z(mB, mV, q2, t0)
    return np.array([1, zq2, zq2**2])

def pole(ff,mres,q2):
    mresdict = {'A0': 0,'A1': 2,'A12': 2,'V': 1,'T1': 1,'T2': 2,'T23': 2}
    m = mres[mresdict[ff]]
    return 1/(1-q2/m**2)

# resonance masses used in BSZ
mres_bsz = {}
mres_bsz['b->d'] = [5.279, 5.324, 5.716];
mres_bsz['b->s'] = [5.366, 5.414, 5.829];

process_dict = {}
process_dict['B->K*'] =    {'B': 'B0', 'V': 'K*0',   'q': 'b->s'}
process_dict['Bs->phi'] =  {'B': 'Bs', 'V': 'phi',   'q': 'b->s'}
process_dict['Bs->K*'] =   {'B': 'Bs', 'V': 'K*0',   'q': 'b->d'}

def ff(process, q2, par, n=2):
    r"""Central value of $B\to V$ form factors in the lattice convention
    and simplified series expansion (SSE) parametrization.

    The lattice convention defines the form factors
    $A_0$, $A_1$, $A_{12}$, $V$, $T_1$, $T_2$, $T_{23}$.

    The SSE defines
    $$F_i(q^2) = P_i(q^2) \sum_k a_k^i \,z(q^2)^k$$
    where $P_i(q^2)=(1-q^2/m_{R,i}^2)^{-1}$ is a simple pole.
    """
    pd = process_dict[process]
    mres = mres_bsz[pd['q']]
    mB = par['m_'+pd['B']]
    mV = par['m_'+pd['V']]
    ff = {}
    for i in ["A0","A1","A12","V","T1","T2","T23"]:
        a = [ par[process + ' SSE ' + i.lower() + '_' + 'a' + str(j)] for j in range(n) ]
        ff[i] = pole(i, mres, q2)*np.dot(a, zs(mB, mV, q2, t0=12.)[:n])
    return ff
