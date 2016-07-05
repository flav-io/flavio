r"""Anomalous dimension matrices for $\Delta F=1$ operators."""

import numpy as np
from math import pi
from flavio.physics.running import running
from flavio.math.functions import zeta

# this file contains the anomalous dimension matrices (ADMs) for the operators
# C_1, C_2 (current-current operators), where C_2 = 1 at LO,
# C_3,4,5,6 (QCD penguins) as defined by Chetyrkin, Misiak & Münz,
# C_7,8^eff (effective (!) dipole operators) with the prefactor {e, g_s)/16pi^2,
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

# (37) of hep-ph/0411071v5
def gamma1_16(f):
    return np.array([[-145/3. + (16*f)/9., -26 + (40*f)/27., -1412/243., -1369/243., 134/243., -35/162.],
 [-45 + (20*f)/3., -28/3., -416/81., 1280/81., 56/81., 35/27.],
 [0, 0, -4468/81., -29129/81. - (52*f)/9., 400/81., 3493/108. - (2*f)/9.],
 [0, 0, -13678/243. + (368*f)/81., -79409/243. + (1334*f)/81., 509/486. - (8*f)/81.,
  13499/648. - (5*f)/27.], [0, 0, -244480/81. - (160*f)/9., -29648/81. - (2200*f)/9.,
  23116/81. + (16*f)/9., 3886/27. + (148*f)/9.], [0, 0, 77600/243. - (1264*f)/81.,
  -28808/243. + (164*f)/81., -20324/243. + (400*f)/81., -21211/162. + (622*f)/27.]])

# (38) of hep-ph/0411071v5
def gamma2_16(f):
    return np.array([[-1927/2. + (257*f)/9. + (40*f**2)/9. + (224 + (160*f)/3.)*zeta(3),
  475/9. + (362*f)/27. - (40*f**2)/27. - (896/3. + (320*f)/9.)*zeta(3),
  269107/13122. - (2288*f)/729. - (1360*zeta(3))/81., -2425817/13122. + (30815*f)/4374. -
   (776*zeta(3))/81., -343783/52488. + (392*f)/729. + (124*zeta(3))/81.,
  -37573/69984. + (35*f)/972. + (100*zeta(3))/27.],
 [307/2. + (361*f)/3. - (20*f**2)/3. - (1344 + 160*f)*zeta(3),
  1298/3. - (76*f)/3. - 224*zeta(3), 69797/2187. + (904*f)/243. + (2720*zeta(3))/27.,
  1457549/8748. - (22067*f)/729. - (2768*zeta(3))/27., -37889/8748. - (28*f)/243. -
   (248*zeta(3))/27., 366919/11664. - (35*f)/162. - (110*zeta(3))/9.],
 [0, 0, -4203068/2187. + (14012*f)/243. - (608*zeta(3))/27.,
  -18422762/2187. + (888605*f)/2916. + (272*f**2)/27. + (39824/27. + 160*f)*zeta(3),
  674281/4374. - (1352*f)/243. - (496*zeta(3))/27., 9284531/11664. - (2798*f)/81. -
   (26*f**2)/27. - (1921/9. + 20*f)*zeta(3)],
 [0, 0, -5875184/6561. + (217892*f)/2187. + (472*f**2)/81. +
   (27520/81. + (1360*f)/9.)*zeta(3), -70274587/13122. + (8860733*f)/17496. -
   (4010*f**2)/729. + (16592/81. + (2512*f)/27.)*zeta(3),
  2951809/52488. - (31175*f)/8748. - (52*f**2)/81. - (3154/81. + (136*f)/9.)*zeta(3),
  3227801/8748. - (105293*f)/11664. - (65*f**2)/54. + (200/27. - (220*f)/9.)*zeta(3)],
 [0, 0, -194951552/2187. + (358672*f)/81. - (2144*f**2)/81. + (87040*zeta(3))/27.,
  -130500332/2187. - (2949616*f)/729. + (3088*f**2)/27. + (238016/27. + 640*f)*zeta(3),
  14732222/2187. - (27428*f)/81. + (272*f**2)/81. - (13984*zeta(3))/27.,
  16521659/2916. + (8081*f)/54. - (316*f**2)/27. - (22420/9. + 200*f)*zeta(3)],
 [0, 0, 162733912/6561. - (2535466*f)/2187. + (17920*f**2)/243. +
   (174208/81. + (12160*f)/9.)*zeta(3), 13286236/6561. - (1826023*f)/4374. -
   (159548*f**2)/729. - (24832/81. + (9440*f)/27.)*zeta(3),
  -22191107/13122. + (395783*f)/4374. - (1720*f**2)/243. - (33832/81. + (1360*f)/9.)*zeta(3),
  -32043361/8748. + (3353393*f)/5832. - (533*f**2)/81. + (9248/27. - (1120*f)/9.)*zeta(3)]])

# (9) of hep-ph/0504194v2
def gamma0_78(f):
    Qd = -1/3.
    return np.array([[32/3.,0],[32/3.*Qd,28/3.]])

# (10) of hep-ph/0504194v2
def gamma1_78(f):
    Qd = -1/3.
    return np.array([[1936/9.- (224*f)/27., 0], [(368/3.- (224*f)/27.)* Qd,
      1456/9.- (61*f)/27.]])

# (11) of hep-ph/0504194v2
def gamma2_78(f):
    Qd = -1/3.
    Qbar = Qsum(f)
    return np.array([[307448/81.- (23776*f)/81.- (352*f**2)/81.,
      0], [-((1600*Qbar)/
        27) + (159872/81.- (17108*f)/81.- (352*f**2)/81.)*Qd,
      268807/81.- (4343*f)/27.- (461*f**2)/81.]])

# (5.9) of hep-ph/0612329
def gamma0_16_78(f):
    Qbar = Qsum(f)
    Qu = 2/3.
    return np.array([[(8/243.) - (4/3.)*Qu, (173/162.)], [-(16/81.) + 8*Qu, (70/27.)],
     [-(176/81.), (14/ 27)], [(88/243.) - (16/81.)*f, (74/81.) - (49/54.)*f],
      [-(6272/ 81), (1736/27.) + 36*f], [(3136/243.) - (160/81.)*f + 48*Qbar, (2372/81.) + (160/27.)*f]])

# (5.10) of hep-ph/0612329
def gamma1_16_78(f):
    Qbar = Qsum(f)
    Qu = 2/3.
    return np.array([[(12614/2187.) - (64/2187.)*f - (374/27.)*Qu + (2/27.)*f*Qu,
     (65867/ 5832) + (431/5832.)*f], [-(2332/729.) + (128/729.)*f + (136/ 9)*Qu - (4/9.)*f*Qu,
      (10577/486.) - (917/972.)*f], [(97876/ 729) - (4352/729.)*f - (112/3.)*Qbar,
       (42524/ 243) - (2398/243.)*f], [-(70376/2187.) - (15788/2187.)*f + (32/ 729)*f**2 - (140/9.)*Qbar,
        -(159718/729.) - (39719/5832.)*f - (253/ 486)*f**2],
         [(1764752/ 729) - (65408/729.)*f - (3136/3.)*Qbar,
          (2281576/ 243) + (140954/243.)*f - 14*f**2],
     [(4193840/ 2187) - (324128/2187.)*f + (896/729.)*f**2 - (1136/9.)*Qbar - (56/ 3)*f*Qbar,
      -(3031517/729.) - (15431/1458.)*f - (6031/486.)*f**2]])

# (5.11) of hep-ph/0612329
def gamma2_16_78(f):
    Qbar = Qsum(f)
    Qu = 2/3.
    g = np.zeros((6,2), dtype=float)
    g[:,0] = np.array([(77506102/ 531441) - (875374/177147.)*f + (560/19683.)*f**2 - (9731/ 162)*Qu + (11045/729.)*f*Qu + (316/729.)*f**2*Qu + (3695/ 486)*Qbar,
    -(15463055/177147.) + (242204/59049.)*f - (1120/ 6561)*f**2 + (55748/27.)*Qu - (33970/243.)*f*Qu - (632/ 243)*f**2*Qu - (3695/81.)*Qbar,
    (102439553/ 177147) - (12273398/59049.)*f + (5824/6561.)*f**2 + (26639/ 81)*Qbar - (8/27.)*f*Qbar,
    -(2493414077/1062882.) - (9901031/ 354294)*f + (243872/59049.)*f**2 - (1184/6561.)*f**3 - (49993/ 972)*Qbar + (305/27.)*f*Qbar,
    (8808397748/ 177147) - (174839456/59049.)*f + (1600/729.)*f**2 - (669694/ 81)*Qbar + (10672/27.)*f*Qbar,
    (7684242746/ 531441) - (351775414/177147.)*f - (479776/59049.)*f**2 - (11456/ 6561)*f**3 + (3950201/243.)*Qbar - (130538/81.)*f*Qbar - (592/ 81)*f**2*Qbar])
    g[:,1] = np.array([-(421272953/1417176.) - (8210077/472392.)*f - (1955/ 6561)*f**2,
    (98548513/ 472392) - (5615165/78732.)*f - (2489/2187.)*f**2,
    (3205172129/ 472392) - (108963529/314928.)*f + (58903/ 4374)*f**2,
    -(6678822461/2834352.) + (127999025/ 1889568)*f + (1699073/157464.)*f**2 + (505/ 4374)*f**3,
    (29013624461/ 118098) - (64260772/19683.)*f - (230962/243.)*f**2 - (148/ 27)*f**3,
    -(72810260309/708588.) + (2545824851/ 472392)*f - (33778271/78732.)*f**2 - (3988/2187.)*f**3])
    #g = g + zeta(3) *
    return np.array([[-(112216/6561.) + (728/729.)*f + (25508/81.)*Qu - (64/81.)*f*Qu - (100/ 27)*Qbar,
    -(953042/2187.) - (10381/486.)*f], [(365696/ 2187) - (1168/243.)*f - (51232/27.)*Qu - (1024/27.)*f*Qu + (200/ 9)*Qbar,
    -(607103/729.) - (1679/81.)*f], [(3508864/ 2187) - (1904/243.)*f - (1984/9.)*Qbar - (64/ 9)*f*Qbar,
    -(1597588/729.) + (13028/81.)*f - (20/ 9)*f**2], [-(1922264/6561.) + (308648/2187.)*f - (1280/ 243)*f**2 + (1010/9.)*Qbar - (200/27.)*f*Qbar,
    (2312684/ 2187) + (128347/729.)*f + (920/81.)*f**2], [(123543040/ 2187) - (207712/243.)*f + (128/27.)*f**2 - (24880/9.)*Qbar - (640/ 9)*f*Qbar,
    -(69359224/729.) - (885356/81.)*f - (5080/ 9)*f**2], [(7699264/ 6561) + (2854976/2187.)*f - (12320/243.)*f**2 - (108584/ 9)*Qbar - (1136/27.)*f*Qbar,
    -(61384768/2187.) - (685472/ 729)*f + (350/81.)*f**2]])
    return g


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

# gammah11CC
def gamma11_12(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[
    ((169)/(9)) , ((100)/(27)) ],[
    ((50)/(3))  ,  -((8)/(3))
    ]] )

# gammah11CP
def gamma11_12_36(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[
    0 , ((254)/(729))  , 0 , 0 ],[
    0 , ((1076)/(243)) , 0 , 0
    ]] )

# gammah11PP
def gamma11_36(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[
     0           , ((11116)/(243))  ,   0          , -((14)/(3)) ],[
    ((280)/(27))  , ((18763)/(729))  , -((28)/(27))  , -((35)/(18)) ],[
     0           , ((111136)/(243)) ,   0          , -((140)/(3)) ],[
    ((2944)/(27)) , ((193312)/(729)) , -((280)/(27)) , -((175)/(9))
    ]] )

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

# gammah11LL  - 2 betaes1 deltaij
def gamma11_910(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[ - 2*176/9. , 16  ],[16 ,  - 2*176/9.]] )

def gamma0_16_9(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array([-32/27., -8/9., -16/9., 32/27., -112/9., 512/27.])

def gamma1_16_9(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array([-2272/729., 1952/243., -6752/243., -2192/729., -84032/243., -37856/729.])

def gamma2_16_9(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array([-1359190/19683. + (6976*zeta(3))/243., -229696/6561. - (3584*zeta(3))/81.,
     -1290092/6561. + (3200*zeta(3))/81., -819971/19683. - (19936*zeta(3))/243.,
     -16821944/6561. + (30464*zeta(3))/81., -17787368/19683. - (286720*zeta(3))/243.])

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

# gamma20QP
def gamma2_Q_36(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[
 ((832)/(243))   , -((4000)/(243))  , -((112)/(243))  , -((70)/(81))    ],[
 ((3376)/(729))  ,  ((6344)/(729))  , -((280)/(729))  ,  ((55)/(486))   ],[
 ((2272)/(243))  , -((72088)/(243)) , -((688)/(243))  , -((1240)/(81))  ],[
 ((45424)/(729)) ,  ((84236)/(729)) , -((3880)/(729)) ,  ((1220)/(243))
]] )

# gamma20QQ
def gamma2_Q_Q(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[
 -((404)/(9))    , -((3077)/(9))   ,  ((32)/(9))    ,  ((1031)/(36))  ],[
 -((2698)/(81))  , -((8035)/(27))  , -((49)/(162))  ,  ((4493)/(216)) ],[
 -((19072)/(9))  , -((14096)/(9))  ,  ((1708)/(9))  ,  ((1622)/(9))   ],[
  ((32288)/(81)) , -((15976)/(27)) , -((6692)/(81)) , -((2437)/(54))
]] )

# gamma20BP
def gamma2_B_36(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[
 -((1576)/(81))    , ((446)/(27))   ,  ((172)/(81))    ,  ((40)/(27))
]] )

# gamma20BB
def gamma2_B_B(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[
 ((325)/(9))
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

# gamma11CQ
def gamma11_12_Q(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[
((2272)/(729))  , ((122)/(81))  , 0 , ((49)/(81)) ],[
-((1952)/(243)) , -((748)/(27)) , 0 , ((82)/(27))
]] )

# gamma11PQ
def gamma11_36_Q(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[
-((23488)/(243))  , ((6280)/(27))   ,  ((112)/(9))  , -((538)/(27)) ],[
 ((31568)/(729))  , ((9481)/(81))   , -((92)/(27))  , -((1012)/(81)) ],[
-((233920)/(243)) , ((68848)/(27))  ,  ((1120)/(9)) , -((5056)/(27)) ],[
 ((352352)/(729)) , ((116680)/(81)) , -((752)/(27)) , -((10147)/(81))
]] )

# gamma11QL
def gamma11_Q_910(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[
-((24352)/(729))    , 0 ],[
 ((54608)/(2187))   , 0 ],[
-((227008)/(729))   , 0 ],[
 ((551648)/(2187))  , 0
]] )

# gamma11BL
def gamma11_B_910(f):
    if f != 5:
        raise  ValueError('Only implemented for 5 flavours')
    return np.array( [[
 -((8)/(9))  ,  0
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
    """
    g=np.zeros((3,2,15,15), dtype=float)
    a = (als/(4*pi))
    ae = (ale/(4*pi))
    # pure QCD
    g[0,0,:6,:6] = a    * gamma0_16(f)
    g[1,0,:6,:6] = a**2 * gamma1_16(f)
    g[2,0,:6,:6] = a**3 * gamma2_16(f)
    g[0,0,6:8,6:8] = a    * gamma0_78(f)
    g[1,0,6:8,6:8] = a**2 * gamma1_78(f)
    g[2,0,6:8,6:8] = a**3 * gamma2_78(f)
    g[0,0,:6,6:8] = a    * gamma0_16_78(f)
    g[1,0,:6,6:8] = a**2 * gamma1_16_78(f)
    g[2,0,:6,6:8] = a**3 * gamma2_16_78(f)
    g[0,0,:6,8] =        gamma0_16_9(f)
    g[1,0,:6,8] = a    * gamma1_16_9(f)
    g[2,0,:6,8] = a**2 * gamma2_16_9(f)
    # QED corrections
    g[0,1,:2,:2] = ae   * gamma01_12(f)
    g[1,1,:2,:2] = a*ae * gamma11_12(f)
    g[1,1,2:6,2:6] = a*ae * gamma11_36(f)
    g[1,1,:2,2:6] = a*ae * gamma11_12_36(f)
    g[0,1,:2,8:10] = ae   * gamma01_12_910(f)
    g[0,1,2:6,8:10] = ae   * gamma01_36_910(f)
    g[0,1,8:10,8:10] = ae   * gamma01_910(f)
    g[1,1,8:10,8:10] = a*ae * gamma11_910(f)
    # EW penguins
    g[1,0,10:14,2:6] =    a**2 * gamma2_Q_36(f)
    g[1,0,10:14,10:14] = a**2 * gamma2_Q_Q(f)
    g[1,0,14,2:6] =      a**2 * gamma2_B_36(f)
    g[1,0,14,14] =       a**2 * gamma2_B_B(f)
    g[0,1,:2,10:14] =    ae   * gamma01_12_Q(f)
    g[1,1,:2,10:14] =    a*ae * gamma11_12_Q(f)
    g[0,1,2:6,10:14] =   ae   * gamma01_36_Q(f)
    g[1,1,2:6,10:14] =   a*ae * gamma11_36_Q(f)
    g[0,0,14,8:10] =            gamma01_B_910(f)
    g[1,0,14,8:10] =     a    * gamma11_B_910(f)
    g[0,0,10:14,8:10] =         gamma01_Q_910(f)
    g[1,0,10:14,8:10] =  a    * gamma11_Q_910(f)
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
    [ C_V, C_S, C_T, C'_V, C'_S ]
    ```
    """
    g=np.zeros((1,1,5,5), dtype=float)
    a = (als/(4*pi))
    g[0,0,2,2] = a*( 8/3 ) # tensor operator
    return g.sum(axis=(0,1))
