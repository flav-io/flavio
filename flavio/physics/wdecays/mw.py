r"""$W$ mass prediction in the presence of new physics."""


from math import log, sqrt
import flavio
from flavio.physics.zdecays import smeftew


def mW_SM(par):
    r"""Two-loop SM prediction for the $W$ pole mass.

    Eq. (6) of arXiv:hep-ph/0311148."""
    dH = log(par['m_h'] / 100)
    dh = (par['m_h'] / 100)**2
    dt = (par['m_t'] / 174.3)**2 - 1
    dZ = par['m_Z'] / 91.1876 - 1
    dalpha = 0  # FIXME
    dalphas = par['alpha_s'] / 0.119 - 1
    m0W = 80.3779
    c1 = 0.05427
    c2 = 0.008931
    c3 = 0.0000882
    c4 = 0.000161
    c5 = 1.070
    c6 = 0.5237
    c7 = 0.0679
    c8 = 0.00179
    c9 = 0.0000664
    c10 = 0.0795
    c11 = 114.9
    return (m0W - c1 * dH - c2 * dH**2 + c3 * dH**4 + c4 * (dh - 1) - c5 * dalpha
            + c6 * dt - c7 * dt**2 - c8 * dH * dt + c9 * dh * dt
            - c10 * dalphas + c11 * dZ)

def dmW_SMEFT(par, C):
    r"""Shift in the $W$ mass due to dimension-6 operators.

    Eq. (2) of arXiv:1606.06502."""
    sh2 = smeftew._sinthetahat2(par)
    sh = sqrt(sh2)
    ch2 = 1 - sh2
    ch = sqrt(ch2)
    Dh = ch * sh / ((ch**2 - sh**2) * 2 * sqrt(2) * par['GF'])
    return Dh * (4 * C['phiWB'] + ch / sh * C['phiD'] + 2 * sh / ch * (C['phil3_11'] + C['phil3_22']) - 2 * sh / ch * C['ll_1221']).real

def mW(wc_obj, par):
    r"""$W$ pole mass."""
    scale = flavio.config['renormalization scale']['wdecays']
    C = wc_obj.get_wcxf(sector='all', scale=scale, par=par,
                        eft='SMEFT', basis='Warsaw')
    dmW = dmW_SMEFT(par, C)
    return mW_SM(par) * (1 - dmW)


_obs_name = "m_W"
_obs = flavio.classes.Observable(_obs_name)
_obs.tex = r"$m_W$"
_obs.set_description(r"$W^\pm$ boson pole mass")
flavio.classes.Prediction(_obs_name, mW)
