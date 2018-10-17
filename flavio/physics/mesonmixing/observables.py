r"""$\Delta F=2$ observables"""


from flavio.physics.mesonmixing import amplitude
from flavio.physics.mesonmixing import common
from flavio.physics import ckm
from flavio.config import config
from math import sqrt, sin
from cmath import phase
from flavio.physics.common import conjugate_par
from flavio.classes import Observable, Prediction


def get_M12_G12(wc_obj, par, meson):
    scale = config['renormalization scale'][meson + ' mixing']
    wc = wc_obj.get_wc(2*common.meson_quark[meson], scale, par)
    M12 = amplitude.M12(par, wc, meson)
    # TODO temporary fix: we don't have a prediction for Gamma12 in the kaon sector
    if meson == 'K0':
        G12 = 0.
    elif meson == 'D0':
        G12 = amplitude.G12_u(par, wc)
    else: # B0 and Bs
        G12 = amplitude.G12_d(par, wc, meson)
    return M12, G12

def DeltaM_12(wc_obj, par, meson):
    r"""Mass difference defined to be $M_1 - M_2$, where the mass eigenstate
    1 is CP-even in the absence of CP violation."""
    M12, G12 = get_M12_G12(wc_obj, par, meson)
    return -common.DeltaM(M12, G12)

def DeltaM_positive(wc_obj, par, meson):
    r"""Mass difference defined to be strictly positive"""
    M12, G12 = get_M12_G12(wc_obj, par, meson)
    return abs(common.DeltaM(M12, G12))

def a_fs(wc_obj, par, meson):
    M12, G12 = get_M12_G12(wc_obj, par, meson)
    return common.a_fs(M12, G12)

def q_over_p(wc_obj, par, meson):
    M12, G12 = get_M12_G12(wc_obj, par, meson)
    return common.q_over_p(M12, G12)

def DeltaGamma_B(wc_obj, par, meson):
    r"""Decay width difference defined as
    $\Delta\Gamma = \Gamma_1 - \Gamma_2$ in the convention where
    $\Delta M = M_2 - M_1$ is positive (and the mass eigenstate
    1 is CP-even in the absence of CP violation), as often used in the
    $B$ system."""
    M12, G12 = get_M12_G12(wc_obj, par, meson)
    DM = common.DeltaM(M12, G12)
    if DM > 0:
        return common.DeltaGamma(M12, G12)
    else:
        return -common.DeltaGamma(M12, G12)

def DeltaGamma_12(wc_obj, par, meson):
    r"""Decay width difference defined as
    $\Delta\Gamma = \Gamma_1 - \Gamma_2$, in the convention where the
    mass eigenstate 1 is CP-even in the absence of CP violation."""
    M12, G12 = get_M12_G12(wc_obj, par, meson)
    return common.DeltaGamma(M12, G12)

def x(wc_obj, par, meson):
    r"""$x=(M_1 - M_2)/\Gamma$ where 1 is CP-even in the CPC limit."""
    return DeltaM_12(wc_obj, par, meson)*par['tau_'+meson]

def y(wc_obj, par, meson):
    r"""$y=(\Gamma_1 - \Gamma_2)/2\Gamma$ where 1 is CP-even in the CPC limit."""
    return DeltaGamma_12(wc_obj, par, meson)*par['tau_'+meson]/2.

def x12(wc_obj, par, meson):
    r"""$x_{12}=2|M_{12}|/\Gamma$."""
    M12, G12 = get_M12_G12(wc_obj, par, meson)
    return 2 * abs(M12) * par['tau_'+meson]

def y12(wc_obj, par, meson):
    r"""$y_{12}=|\Gamma_{12}|/\Gamma$."""
    M12, G12 = get_M12_G12(wc_obj, par, meson)
    return abs(G12) * par['tau_'+meson]

def x12Im(wc_obj, par, meson):
    r"""$x_{12}^\text{Im}=\sin\phi_{12}$."""
    return x12(wc_obj, par, meson) * sin(phi12(wc_obj, par, meson))

def phi12(wc_obj, par, meson):
    r"""$\phi_{12}=\text{arg}(-M_{12}/\Gamma_{12})"""
    M12, G12 = get_M12_G12(wc_obj, par, meson)
    return phase(M12 / G12)

def epsK(wc_obj, par):
    M12, G12 = get_M12_G12(wc_obj, par, 'K0')
    keps =  par['kappa_epsilon']
    DMK =  par['DeltaM_K0']
    return keps * M12.imag / DMK / sqrt(2)

def amplitude_BJpsiK(par):
    xi_c = ckm.xi('c', 'bd')(par) # V_cb V_cd*
    return xi_c

def amplitude_Bspsiphi(par):
    xi_c = ckm.xi('c', 'bs')(par) # V_cb V_cs*
    return xi_c

def S(wc_obj, par, meson, amplitude, etaCP):
    M12, G12 = get_M12_G12(wc_obj, par, meson)
    qp = common.q_over_p(M12, G12)
    DM = common.DeltaM(M12, G12)
    if DM < 0:
        qp = -qp  # switch the sign of q/p to keep DeltaM > 0
    A = amplitude(par)
    A_bar = amplitude(conjugate_par(par))
    xi = etaCP * qp * A / A_bar
    return -2*xi.imag / ( 1 + abs(xi)**2 )

def S_BJpsiK(wc_obj, par):
    return S(wc_obj, par, 'B0', amplitude_BJpsiK, etaCP=-1)

def S_Bspsiphi(wc_obj, par):
    return S(wc_obj, par, 'Bs', amplitude_Bspsiphi, etaCP=+1)


# Observable and Prediction instances

o = Observable('DeltaM_s')
o.set_description(r"Mass difference in the $B_s$-$\bar B_s$ system")
o.tex = r"$\Delta M_s$"
o.add_taxonomy(r'Process :: Meson-antimeson mixing ::  $B_s$-$\bar B_s$ mixing')
Prediction('DeltaM_s', lambda wc_obj, par: DeltaM_positive(wc_obj, par, 'Bs'))

o = Observable('DeltaM_d')
o.set_description(r"Mass difference in the $B^0$-$\bar B^0$ system")
o.tex = r"$\Delta M_d$"
o.add_taxonomy(r'Process :: Meson-antimeson mixing ::  $B^0$-$\bar B^0$ mixing')
Prediction('DeltaM_d', lambda wc_obj, par: DeltaM_positive(wc_obj, par, 'B0'))

o = Observable('a_fs_s')
o.set_description(r"CP asymmetry in flavour-specific $B_s$ decays")
o.tex = r"$a_\text{fs}^s$"
o.add_taxonomy(r'Process :: Meson-antimeson mixing ::  $B_s$-$\bar B_s$ mixing')
Prediction('a_fs_s', lambda wc_obj, par: a_fs(wc_obj, par, 'Bs'))

o = Observable('a_fs_d')
o.set_description(r"CP asymmetry in flavour-specific $B^0$ decays")
o.tex = r"$a_\text{fs}^d$"
o.add_taxonomy(r'Process :: Meson-antimeson mixing ::  $B^0$-$\bar B^0$ mixing')
Prediction('a_fs_d', lambda wc_obj, par: a_fs(wc_obj, par, 'B0'))

o = Observable('DeltaGamma_s')
o.set_description(r"Decay width difference in the $B_s$-$\bar B_s$ system")
o.tex = r"$\Delta\Gamma_s$"
o.add_taxonomy(r'Process :: Meson-antimeson mixing ::  $B_s$-$\bar B_s$ mixing')
Prediction('DeltaGamma_s', lambda wc_obj, par: DeltaGamma_B(wc_obj, par, 'Bs'))

o = Observable('DeltaGamma_d')
o.set_description(r"Decay width difference in the $B^0$-$\bar B^0$ system")
o.tex = r"$\Delta\Gamma_d$"
o.add_taxonomy(r'Process :: Meson-antimeson mixing ::  $B^0$-$\bar B^0$ mixing')
Prediction('DeltaGamma_d', lambda wc_obj, par: DeltaGamma_B(wc_obj, par, 'B0'))

o = Observable('eps_K')
o.set_description(r"Indirect CP violation parameter in the $K^0$-$\bar K^0$ system")
o.tex = r"$\vert\epsilon_K\vert$"
o.add_taxonomy(r'Process :: Meson-antimeson mixing ::  $K^0$-$\bar K^0$ mixing')
Prediction('eps_K', epsK)

o = Observable('S_psiK')
o.set_description(r"Mixing induced CP asymmetry in $B^0\to J/\psi K_S$")
o.tex = r"$S_{\psi K_S}$"
o.add_taxonomy(r'Process :: Meson-antimeson mixing ::  $B^0$-$\bar B^0$ mixing')
o.add_taxonomy(r'Process :: $b$ hadron decays :: Non-leptonic decays :: $B\to VP$')
Prediction('S_psiK', S_BJpsiK)

o = Observable('S_psiphi')
o.set_description(r"Mixing induced CP asymmetry in $B_s\to J/\psi \phi$")
o.tex = r"$S_{\psi\phi}$"
o.add_taxonomy(r'Process :: Meson-antimeson mixing ::  $B_s$-$\bar B_s$ mixing')
o.add_taxonomy(r'Process :: $b$ hadron decays :: Non-leptonic decays :: $B\to VV$')
Prediction('S_psiphi', S_Bspsiphi)

o = Observable('x_D')
o.set_description(r"Normalized mass difference in the $D^0$-$\bar D^0$ system")
o.tex = r"$x_D$"
o.add_taxonomy(r'Process :: Meson-antimeson mixing ::  $D^0$-$\bar D^0$ mixing')
Prediction('x_D', lambda wc_obj, par: x(wc_obj, par, 'D0'))

o = Observable('y_D')
o.set_description(r"Normalized width difference in the $D^0$-$\bar D^0$ system")
o.tex = r"$y_D$"
o.add_taxonomy(r'Process :: Meson-antimeson mixing ::  $D^0$-$\bar D^0$ mixing')
Prediction('y_D', lambda wc_obj, par: y(wc_obj, par, 'D0'))

o = Observable('phi_D')
o.set_description(r"Phase of $q/p$ in the $D^0$-$\bar D^0$ system")
o.tex = r"$\phi_D$"
o.add_taxonomy(r'Process :: Meson-antimeson mixing ::  $D^0$-$\bar D^0$ mixing')
Prediction('phi_D', lambda wc_obj, par: phase(-q_over_p(wc_obj, par, 'D0')))

o = Observable('q/p_D')
o.set_description(r"Absolute value of $q/p$ in the $D^0$-$\bar D^0$ system")
o.tex = r"$\vert q/p\vert_D$"
o.add_taxonomy(r'Process :: Meson-antimeson mixing ::  $D^0$-$\bar D^0$ mixing')
Prediction('q/p_D', lambda wc_obj, par: abs(q_over_p(wc_obj, par, 'D0')))

o = Observable('x12_D')
o.set_description(r"Normalized absolute value of the dispersive mixing amplitude in the $D^0$-$\bar D^0$ system")
o.tex = r"$x_{12}^D$"
o.add_taxonomy(r'Process :: Meson-antimeson mixing ::  $D^0$-$\bar D^0$ mixing')
Prediction('x12_D', lambda wc_obj, par: x12(wc_obj, par, 'D0'))

o = Observable('y12_D')
o.set_description(r"Normalized absolute value of the absorptive mixing amplitude in the $D^0$-$\bar D^0$ system")
o.tex = r"$y_{12}^D$"
o.add_taxonomy(r'Process :: Meson-antimeson mixing ::  $D^0$-$\bar D^0$ mixing')
Prediction('y12_D', lambda wc_obj, par: y12(wc_obj, par, 'D0'))

o = Observable('phi12_D')
o.set_description(r"$D^0$-$\bar D^0$ mixing phase")
o.tex = r"$\phi_{12}^D$"
o.add_taxonomy(r'Process :: Meson-antimeson mixing ::  $D^0$-$\bar D^0$ mixing')
Prediction('phi12_D', lambda wc_obj, par: phi12(wc_obj, par, 'D0'))

o = Observable('x12Im_D')
o.set_description(r"Normalized imaginary part of the dispersive mixing amplitude in the $D^0$-$\bar D^0$ system")
o.tex = r"$x_{12}^{\text{Im},D}$"
o.add_taxonomy(r'Process :: Meson-antimeson mixing ::  $D^0$-$\bar D^0$ mixing')
Prediction('x12Im_D', lambda wc_obj, par: x12Im(wc_obj, par, 'D0'))
