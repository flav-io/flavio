from math import sqrt
import numpy as np
from flavio.physics.bdecays.formfactors.common import z
from flavio.config import config
from functools import lru_cache

@lru_cache(maxsize=config['settings']['cache size'])
def zs(mB, mV, q2, t0):
    zq2 = z(mB, mV, q2, t0)
    z0 = z(mB, mV, 0, t0)
    return np.array([1, zq2-z0, (zq2-z0)**2])

def pole(ff,mres,q2):
    mresdict = {'A0': 0,'A1': 2,'A12': 2,'V': 1,'T1': 1,'T2': 2,'T23': 2}
    m = mres[mresdict[ff]]
    return 1/(1-q2/m**2)

# resonance masses used in arXiv:1503.05534v1
mres_bsz = {}
mres_bsz['b->d'] = [5.279, 5.324, 5.716];
mres_bsz['b->s'] = [5.367, 5.415, 5.830];

process_dict = {}
process_dict['B->K*'] =    {'B': 'B0', 'V': 'K*0',   'q': 'b->s'}
process_dict['B->rho'] =   {'B': 'B0', 'V': 'rho0',  'q': 'b->d'}
process_dict['B->omega'] = {'B': 'B0', 'V': 'omega', 'q': 'b->d'}
process_dict['Bs->phi'] =  {'B': 'Bs', 'V': 'phi',   'q': 'b->s'}
process_dict['Bs->K*'] =   {'B': 'Bs', 'V': 'K*0',   'q': 'b->d'}

def ff(process, q2, par, n=2):
    r"""Central value of $B\to V$ form factors in the lattice convention
    and BSZ parametrization.

    The lattice convention defines the form factors
    $A_0$, $A_1$, $A_{12}$, $V$, $T_1$, $T_2$, $T_{23}$.

    The BSZ parametrization defines

    $$F_i(q^2) = P_i(q^2) \sum_k a_k^i \,\left[z(q^2)-z(0)\right]^k$$

    where $P_i(q^2)=(1-q^2/m_{R,i}^2)^{-1}$ is a simple pole.
    """
    pd = process_dict[process]
    mres = mres_bsz[pd['q']]
    mB = par['m_'+pd['B']]
    mV = par['m_'+pd['V']]
    ff = {}
    # setting a0_A0 and a0_T2 according to the exact kinematical relations,
    # cf. eq. (16) of arXiv:1503.05534
    par_copy = par.copy()
    par_prefix = process + ' BSZ'
    par_copy[par_prefix + ' a0_A0'] = 8*mB*mV/(mB**2-mV**2)*par_copy[par_prefix + ' a0_A12']
    par_copy[par_prefix + ' a0_T2'] = par_copy[par_prefix + ' a0_T1']
    for i in ["A0","A1","A12","V","T1","T2","T23"]:
        a = [ par_copy[par_prefix + ' a' + str(j) + '_' + i] for j in range(n) ]
        ff[i] = pole(i, mres, q2)*np.dot(a, zs(mB, mV, q2, t0=None)[:n])
    return ff
