from flavio.physics.mesonmixing import amplitude
from flavio.physics.mesonmixing.common import wcnp_dict, meson_quark
from flavio.config import config
from math import sqrt
from cmath import phase

def get_M12(wc_obj, par, meson):
    scale = config['mesonmixing']['scale_mix_' + meson]
    wc = wcnp_dict(wc_obj, 'df2_' + meson_quark[meson], scale, par)
    M12 = amplitude.M12_d(par, wc, meson)
    return M12

def DeltaM(wc_obj, par, meson):
    M12 = get_M12(wc_obj, par, meson)
    return 2*abs(M12)

def phi(wc_obj, par, meson):
    M12 = get_M12(wc_obj, par, meson)
    return phase(M12)

def epsK(wc_obj, par):
    M12 = get_M12(wc_obj, par, 'K0')
    keps =  par['kappa_epsilon']
    DMK =  par[('DeltaM','K0')]
    return keps * M12.imag / DMK / sqrt(2)
