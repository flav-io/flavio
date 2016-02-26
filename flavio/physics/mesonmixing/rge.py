"""Anomalous dimension matrix for meson-antimeson mixing"""

import numpy as np
from math import pi
from flavio.physics.running import running

N = 3 # QCD number of colours

def gamma_df2_dict(f):
    gamma = {}
    gamma['SLL'] = np.zeros((2,2,2))
    gamma['VLL'] = np.zeros((2))
    gamma['LR'] = np.zeros((2,2,2))

    # TakeN * from hep-ph/0005183v1

    # (2.18)
    gamma['SLL'][0,0,0] = -6 * N +6 +(6/N)
    gamma['SLL'][0,0,1] = (1/2) -(1/N)
    gamma['SLL'][0,1,0] = -24 -(48/N)
    gamma['SLL'][0,1,1] = 2 * N + 6 -(2/N)

    # (2.20)
    gamma['SLL'][1,0,0] = -(203/6) * N**2 +(107/3) * N +(136/3) -(12/N) -((107)/(2 * N**2)) +(10/3) * N * f -(2/3) * f -(10/(3 * N)) * f
    gamma['SLL'][1,0,1] = -(1/36) * N -(31/9) +(9/N) -((4)/(N**2)) -(1/18) * f +(1/(9 * N)) * f
    gamma['SLL'][1,1,0] = -(364/3) * N -(704/3) -(208/N) -((320)/(N**2)) +(136/3) * f + (176/(3 * N)) * f
    gamma['SLL'][1,1,1] = (343/18) * N**2 +21 * N -(188/9) +(44/N) +((21)/(2 * N**2)) -(26/9) * N * f -6 * f +(2/(9 * N)) * f

    # (2.21)
    gamma['VLL'][0] = 6 - (6/N)
    gamma['VLL'][1] = -(19/6) * N -(22/3) +(39/N) -((57)/(2 * N**2)) + 2/3.*f-2/(3.*N)*f

    # (2.23)
    gamma['LR'][0,0,0] = (6/N)
    gamma['LR'][0,0,1] = 12
    gamma['LR'][0,1,0] = 0
    gamma['LR'][0,1,1] = - 6 * N + (6/N)

    # (2.24)
    gamma['LR'][1,0,0] = (137/6) + ((15)/(2 * N**2)) - (22/(3 * N)) * f
    gamma['LR'][1,0,1] = (200/3) * N -(6/N) -(44/3) * f
    gamma['LR'][1,1,0] = (71/4) * N +(9/N) -2 * f
    gamma['LR'][1,1,1] = -(203/6) * N**2 +(479/6) +(15/(2 * N**2)) +(10/3) * N * f -(22/(3 * N)) * f
    return gamma

def gamma_df2_array(f, alpha_s):
    gamma_dict = gamma_df2_dict(f)
    a =  alpha_s/(4.*pi)
    # C = alpha_s/(4pi) * ( C^0 + alpha_s/(4pi) * C^1 )
    gamma_dict_nlo = {k: a*c[0]+a**2*c[1] for k,c in gamma_dict.items()}
    # Basis: 0: VLL, 1: SLL, 2: TLL, 3: VRR, 4: SRR, 5: TRR, 6: VLR, 7: SLR
    gamma=np.zeros((8,8))
    gamma[0,0] = gamma_dict_nlo['VLL']
    gamma[1,1] = gamma_dict_nlo['SLL'][0,0] # SLL
    gamma[2,2] = gamma_dict_nlo['SLL'][1,1] # TLL
    gamma[1,2] = gamma_dict_nlo['SLL'][0,1] # SLL-TLL
    gamma[2,1] = gamma_dict_nlo['SLL'][1,0] # TLL-SLL
    gamma[6,6] = gamma_dict_nlo['LR'][0,0] # VLR
    gamma[7,7] = gamma_dict_nlo['LR'][1,1] # SLR
    gamma[6,7] = gamma_dict_nlo['LR'][0,1] # VLR-SLR
    gamma[7,6] = gamma_dict_nlo['LR'][1,0] # SLR-VLR
    # and due to QCD parity invariance:
    gamma[3:6,3:6] = gamma[0:3,0:3] # XRR = XLL
    return gamma


def gamma_df2(c, alpha_s, mu, nf):
    r"""RHS of the RGE for $\Delta F=2$ Wilson coefficients written in the form
    $$\frac{d}{d\mu} \vec C = \gamma^T \vec C / \mu \equiv \vec R$$
    """
    gamma = gamma_df2_array(nf, alpha_s)
    return np.dot(gamma.T, c)/mu

def run_wc_df2(par, c_in, scale_in, scale_out):
    adm = lambda nf, alpha_s, alpha_e: gamma_df2_array(nf, alpha_s)
    return running.get_wilson(par, c_in, running.make_wilson_rge_derivative(adm), scale_in, scale_out)
