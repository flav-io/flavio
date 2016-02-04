from flavio.physics.mesonmixing import amplitude
from flavio.physics.mesonmixing import common
from flavio.physics import ckm
from flavio.config import config
from math import sqrt, sin
from cmath import phase
from flavio.physics.common import conjugate_par

def get_M12_G12(wc_obj, par, meson):
    scale = config['renormalization scale'][meson + ' mixing']
    wc = wc_obj.get_wc(2*common.meson_quark[meson], scale, par)
    M12 = amplitude.M12_d(par, wc, meson)
    G12 = amplitude.G12_d(par, wc, meson)
    return M12, G12

def DeltaM(wc_obj, par, meson):
    M12, G12 = get_M12_G12(wc_obj, par, meson)
    return common.DeltaM(M12, G12)

def a_fs(wc_obj, par, meson):
    M12, G12 = get_M12_G12(wc_obj, par, meson)
    return common.a_fs(M12, G12)

def q_over_p(wc_obj, par, meson):
    M12, G12 = get_M12_G12(wc_obj, par, meson)
    return common.q_over_p(M12, G12)

def DeltaGamma(wc_obj, par, meson):
    M12, G12 = get_M12_G12(wc_obj, par, meson)
    return common.DeltaGamma(M12, G12)


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

def S(wc_obj, par, meson, amplitude):
    M12, G12 = get_M12_G12(wc_obj, par, meson)
    qp = common.q_over_p(M12, G12)
    A = amplitude(par)
    A_bar = amplitude(conjugate_par(par))
    xi = qp * A / A_bar
    return 2*xi.imag / ( 1 + abs(xi)**2 )

def S_BJpsiK(wc_obj, par):
    return S(wc_obj, par, 'B0', amplitude_BJpsiK)

def S_Bspsiphi(wc_obj, par):
    return S(wc_obj, par, 'Bs', amplitude_Bspsiphi)
