r"""Anomalous dimension matrices for $\Delta F=1$ operators."""

import numpy as np
from math import pi
from flavio.physics.running import running
from flavio.math.functions import zeta

# this file contains the one-loop anomalous dimension matrices (ADMs) for the operators
# C_1, C_2 (current-current operators), where C_2 = 1 at LO,
# C_3,4,5,6 (QCD penguins) as defined by Chetyrkin, Misiak & Münz,
# C_7,8  with the prefactor {e, g_s)/16pi^2,
# (note that C7,8eff vs C7,8 only makes a differences from two loops!)
# C_9 (semi-leptonic vector operator, not the effective one!) with prefactor e^2/16pi^2
# C_10 (semi-leptonic axial vector operator) with prefactor e^2/16pi^2
# C_3Q-C_6Q, C_b (electroweak penguins) as defined by Huber/Lunghi/Misiak/Wyler.

def Qsum(f):
    r"""Sum of quark charges for $n_f$ active quark flavours"""
    d = {
    1: 2/3., # u
    2: 1/3., # ud
    3: 0, # uds
    4: 2/3., # udsc
    5: 1/3., # udscb
    6: 1, # udscbt
    }
    return d[f]

# (36) of hep-ph/0411071v5
def gamma0_16(f):
    return np.array([[-4, 8/3., 0, -2/9., 0, 0], [12, 0, 0, 4/3., 0, 0], [0, 0, 0, -52/3., 0, 2],
 [0, 0, -40/9., -160/9. + (4*f)/3., 4/9., 5/6.], [0, 0, 0, -256/3., 0, 20],
 [0, 0, -256/9., -544/9. + (40*f)/3., 40/9., -2/3.]])

# (9) of hep-ph/0504194v2
def gamma0_78(f):
    Qd = -1/3.
    return np.array([[32/3.,0],[32/3.*Qd,28/3.]])

# (5.9) of hep-ph/0612329
def gamma0_16_78(f):
    Qbar = Qsum(f)
    Qu = 2/3.
    return np.array([[(8/243.) - (4/3.)*Qu, (173/162.)], [-(16/81.) + 8*Qu, (70/27.)],
     [-(176/81.), (14/ 27)], [(88/243.) - (16/81.)*f, (74/81.) - (49/54.)*f],
      [-(6272/ 81), (1736/27.) + 36*f], [(3136/243.) - (160/81.)*f + 48*Qbar, (2372/81.) + (160/27.)*f]])


# The following ADMs are taken from hep-ph/0512066v4.
# Note that this code uses a different normalization for C_9,10 (namely with an
# additional alpha/4pi in the definition of the operator, as conventionally
# done in pheno analyses), so the gamma^{m,n} of this paper become gamma^{m,n-1}
# in the case of C_9,10 below. Also, this changes the self-mixing of C_9,10
# (it vanishes up to QED corrections.)
# For C_1-6, I change gamma^{m,n} to
# gamma^{m-1,n}. Sorry.


# gammah01CC
def gamma01_12(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[-((8)/(3)) , 0 ],[0 ,  -((8)/(3))]] )


# gammah02CL
def gamma01_12_910(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[ -((11680)/(2187)) , -((416)/(81)) ],[
     -((2920)/(729))   ,  -((104)/(27))
    ]] )

# gammah02PL
def gamma01_36_910(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[-((39752)/(729))  , -((136)/(27)) ],[
     ((1024)/(2187))  , -((448)/(81)) ],[
    -((381344)/(729)) , -((15616)/(27)) ],[
     ((24832)/(2187)) , -((7936)/(81))
    ]] )

# gammah01LL - 2 betae0 deltaij
def gamma01_910(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[   8 - 2*80/9. ,  -4 ],[ -4 ,   - 2*80/9.]] )

def gamma0_16_9(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array([-32/27., -8/9., -16/9., 32/27., -112/9., 512/27.])

def gamma1_16_9(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array([-2272/729., 1952/243., -6752/243., -2192/729., -84032/243., -37856/729.])

# gamma10QP
def gamma10QP(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[
 0 ,    -((8)/(9)) ,          0 ,          0 ],[
 0 ,   ((16)/(27)) ,          0 ,          0 ],[
 0 ,  -((128)/(9)) ,          0 ,          0 ],[
 0 ,  ((184)/(27)) ,          0 ,          0
]] )

# gamma10QQ
def gamma10QQ(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[
           0 ,          -20 ,          0 ,          2 ],[
  -((40)/(9)) ,   -((52)/(3)) ,   ((4)/(9)) ,   ((5)/(6)) ],[
           0 ,         -128 ,          0 ,         20 ],[
 -((256)/(9)) ,  -((160)/(3)) ,  ((40)/(9)) ,  -((2)/(3))
]] )

def gamma10BB(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return [ 4 ],

def gamma10BP(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[
0     ,    ((4)/(3)) ,      0    ,     0
]] )

# gamma01BL
def gamma01_B_910(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[
  ((16)/(9)) , 0
]] )

# gamma01CQ
def gamma01_12_Q(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[
   ((32)/(27)) ,           0 ,           0 ,           0 ],[
     ((8)/(9)) ,           0 ,           0 ,           0
]] )

# gamma01PQ
def gamma01_36_Q(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[
    ((76)/(9)) ,           0 ,   -((2)/(3)) ,           0 ],[
  -((32)/(27)) ,   ((20)/(3)) ,           0 ,   -((2)/(3)) ],[
   ((496)/(9)) ,           0 ,  -((20)/(3)) ,           0 ],[
 -((512)/(27)) ,  ((128)/(3)) ,           0 ,  -((20)/(3))
]] )

# gamma01QL
def gamma01_Q_910(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[
  -((272)/(27)) ,         0],[
   -((32)/(81)) ,         0],[
 -((2768)/(27)) ,         0],[
  -((512)/(81)) ,         0
]] )


# putting everything together

def gamma_all_orders(f, als, ale):
    r"""Returns the ADM in the basis

    ```
    [ C_1, C_2, C_3, C_4, C_5, C_6,
    C_7^eff, C_8^eff,
    C_9, C_10,
    C_3^Q, C_4^Q, C_5^Q, C_6^Q,
    Cb ]
    ```

    where all operators are defined as in hep-ph/0512066 *except*
    $C_{9,10}$, which are defined with an additional $\alpha/4\pi$ prefactor.

    Output is a (3,2,15,15)-array where the last 2 axes are the 15 Wilson
    coefficients and the first to axes are the QCD and QED orders; 0 is LO,
    1 is NLO, 2 is NNLO (only for QCD).

    NB: NLO and NNLO contributions have been removed in flavio v0.25!
    """
    g=np.zeros((3,2,15,15), dtype=float)
    a = (als/(4*pi))
    ae = (ale/(4*pi))
    # pure QCD
    g[0,0,:6,:6] = 0*a    * gamma0_16(f)
    # g[1,0,:6,:6] = a**2 * gamma1_16(f)
    # g[2,0,:6,:6] = a**3 * gamma2_16(f)
    g[0,0,6:8,6:8] = a    * gamma0_78(f)
    # g[1,0,6:8,6:8] = a**2 * gamma1_78(f)
    # g[2,0,6:8,6:8] = a**3 * gamma2_78(f)
    g[0,0,:6,6:8] = a    * gamma0_16_78(f)
    # g[1,0,:6,6:8] = a**2 * gamma1_16_78(f)
    # g[2,0,:6,6:8] = a**3 * gamma2_16_78(f)
    g[0,0,:6,8] =        gamma0_16_9(f)
    # g[1,0,:6,8] = a    * gamma1_16_9(f)
    # g[2,0,:6,8] = a**2 * gamma2_16_9(f)
    # QED corrections
    g[0,1,:2,:2] = ae   * gamma01_12(f)
    # g[1,1,:2,:2] = a*ae * gamma11_12(f)
    # g[1,1,2:6,2:6] = a*ae * gamma11_36(f)
    # g[1,1,:2,2:6] = a*ae * gamma11_12_36(f)
    g[0,1,:2,8:10] = ae   * gamma01_12_910(f)
    g[0,1,2:6,8:10] = ae   * gamma01_36_910(f)
    g[0,1,8:10,8:10] = ae   * gamma01_910(f)
    # g[1,1,8:10,8:10] = a*ae * gamma11_910(f)
    # EW penguins
    # g[1,0,10:14,2:6] =    a**2 * gamma2_Q_36(f)
    # g[1,0,10:14,10:14] = a**2 * gamma2_Q_Q(f)
    # g[1,0,14,2:6] =      a**2 * gamma2_B_36(f)
    # g[1,0,14,14] =       a**2 * gamma2_B_B(f)
    g[0,1,:2,10:14] =    ae   * gamma01_12_Q(f)
    # g[1,1,:2,10:14] =    a*ae * gamma11_12_Q(f)
    g[0,1,2:6,10:14] =   ae   * gamma01_36_Q(f)
    # g[1,1,2:6,10:14] =   a*ae * gamma11_36_Q(f)
    g[0,0,14,8:10] =            gamma01_B_910(f)
    # g[1,0,14,8:10] =     a    * gamma11_B_910(f)
    g[0,0,10:14,8:10] =         gamma01_Q_910(f)
    # g[1,0,10:14,8:10] =  a    * gamma11_Q_910(f)
    return g

def gamma_all(f, als, ale, n_s=3, n_e=2):
    r"""Returns the ADM in the basis

    ```
    [ C_1, C_2, C_3, C_4, C_5, C_6,
    C_7^eff, C_8^eff,
    C_9, C_10,
    C_3^Q, C_4^Q, C_5^Q, C_6^Q,
    Cb ]
    ```

    where all operators are defined as in hep-ph/0512066 *except*
    $C_{9,10}$, which are defined with an additional $\alpha/4\pi$ prefactor.

    The arguments `n_s` and `n_e` can be used to limit the QCD and QED orders taken
    into account. 1 is LO, 2 is NLO, 3 is NNLO (only for QCD).
    """
    g = gamma_all_orders(f, als, ale)
    return g[:n_s,:n_e,:,:].sum(axis=(0,1))

def run_wc_df1(par, c_in, scale_in, scale_out):
    adm = gamma_all
    return running.get_wilson(par, c_in, running.make_wilson_rge_derivative(adm), scale_in, scale_out)

def gamma_fccc(als, ale):
    r"""Returns the ADMs for $d_i\to d_j\ell\nu decays in the basis

    ```
    [ C_VL, C_SR, C_T, C_VR, C_SL ]
    ```
    """
    g=np.zeros((1,1,5,5), dtype=float)
    a = (als/(4*pi))
    g[0,0,1,1] = a*( -8 ) # scalar operator
    g[0,0,4,4] = a*( -8 ) # scalar operator
    g[0,0,2,2] = a*( 8/3 ) # tensor operator
    return g.sum(axis=(0,1))
