r"""Functions for Bhabha scattering: $e^+ e^-\to e^+ e^- """
# Written by Ben Allanach

from math import pi, sqrt
from flavio.physics.zdecays.smeftew import gV_SM, gA_SM, _QN
from flavio.physics.common import add_dict
from flavio.classes import Observable, Prediction
from flavio.physics import ckm as ckm_flavio
import scipy
from scipy.integrate import quad
import numpy as np
import flavio.physics.zdecays.smeftew as smeftew

# Kronecker delta
def delta(a, b):
    return int(a == b)

# predicted ratio to SM with SMEFT operators of e+e-->mumu for LEP2 energy E and family fam total cross-section. afb should be true for forward-backward asymmetry, whereas it should be false for the total cross-section. cthmin and chtmax are the minimum and maximum values of cosine of the scattering angle in the current bin. Programmed by BCA 24/4/23
def ee_ll(C, par, E, cthmin, cthmax):
    # Check energy E is correct
    if (E != 182.7 and E != 188.6 and E != 191.6 and E != 195.5 and
        E != 199.5 and E!= 201.6 and E!= 204.9 and E!= 206.6):
        raise ValueError('ee_ee called with incorrect LEP2 energy {} GeV.'.format(E))
    if (cthmin != -0.90 and cthmin != -0.72 and cthmin != -0.54 and
        cthmin != -0.36 and cthmin != -0.18 and cthmin != 0.00 and
        cthmin != 0.18 and cthmin != 0.27 and cthmin != 0.36 and
        cthmin != 0.45 and cthmin != 0.54 and cthmin != 0.63 and
        cthmin != 0.72 and cthmin != 0.81 and cthmin != 0.90):
        raise ValueError('ee_ee called with incorrect cthmin {}'.format(cthmin))
    if (cthmax != -0.90 and cthmax != -0.72 and cthmax != -0.54 and
        cthmax != -0.36 and cthmax != -0.18 and cthmax != 0.00 and
        cthmax != 0.18 and cthmax != 0.27 and cthmax != 0.36 and
        cthmax != 0.45 and cthmax != 0.54 and cthmax != 0.63 and
        cthmax != 0.72 and cthmax != 0.81 and cthmax != 0.90):
        raise ValueError('ee_ee called with incorrect cthmax {}'.format(cthmax))
    s = E * E
    ssq = s**2
    MZ = par['m_Z']
    MZsq = MZ**2
    MZ4 = MZsq**2
    MZ6 = MZ4 * MZsq
    MZ8 = MZ4**2
    MZsqmsSq = (MZsq - s)**2
    gammaZ = 1 / par['tau_Z']
    gammaZsq = gammaZ**2
    gammaZ4 = gammaZsq4
    gammaZ6 = gammaZsq4 * gammaZsq
    GF = par['GF']
    alpha = par['alpha_e']
    s2w   = par['s2w']
    gzeL  = -0.5 + s2w
    gzeR  = s2w
    eSq   = 4 * pi * alpha1
    gLsq  = eSq / s2w
    gYsq  = gLsq * s2w / (1. - s2w)
    g_cw  = sqrt(gLsq + gYsq)
    sigma_tot    = 0
    sigma_tot_SM = 0
    gVe  = g_cw * gV_SM('e', par) 
    gAe  = g_cw * gA_SM('e', par)
    # f is proportional to d sigma / d (c=cos theta) integrated over c.
    # c is cos(theta)L    
    # X is 1 or 2, so is Y (meaning L or R).
    # Z is 1 for including new physics, 0 for SM
    def f(c, X, Y, Z):
        t = -s * 0.5 * (1.0 - c)
        u = -s * 0.5 * (1.0 + c)
        qed = 2 * eSq**2 * (
            (u**2 + t**2) / t**2 + (u**2 + t**2) / s**2 + 2 * u**2 / (s * t)
            )
        # SM versions 
        gex = (gVe + gAe) * delta(X, 2) + (gVe - gAe) * delta(X, 1)
        gey = (gVe + gAe) * delta(Y, 2) + (gVe - gAe) * delta(Y, 1)
        tXY = gex * gey / (t - MZ**2 + j * gammaZ * MZ)
        sXY = gex * gey / (t - MZ**2 + j * gammaZ * MZ)
        CXY = 0
        CYX = 0
        if (Z == 1): # add SMEFT contributions
            gex =+ (smeftew.d_gVl('e', 'e', par, C) + smeftew.d_gAl('e', 'e', par, C)) * delta(X, 2) + (smeftew.d_gVl('e', 'e', par, C) - smeftew.d_gAl('e', 'e', par, C)) * delta(X, 1)
            gex =+ (smeftew.d_gVl('e', 'e', par, C) + smeftew.d_gAl('e', 'e', par, C)) * delta(X, 2) + (smeftew.d_gVl('e', 'e', par, C) - smeftew.d_gAl('e', 'e', par, C)) * delta(X, 1)
            gey =+ geySM + (smeftew.d_gVl(l_type, l_type, par, C) + smeftew.d_gAl(l_type, l_type, par, C)) * delta(Y, 2) + (smeftew.d_gVl(l_type, l_type, par, C) - smeftew.d_gAl(l_type, l_type, par, C)) * delta(Y, 1)
            if (X == 1 and Y == 1):
                CXY = C[f'll_1111']
                CYX = C[f'll_1111']
            elif ((X == 1 and Y == 2) or (X == 2 and Y == 1)):
                CXY = C[f'le_1111']
                CYX = C[f'le_1111']
            elif (X == 2 and Y == 2):
                CXY = C[f'ee_1111']
                CYX = C[f'ee_1111']
            else:
                raise ValueError('f called with incorrect X,Y {}'.format(X,Y))
            tXY =+ CXY
            sXY =+ CXY
        gexgey = gex * gey
        deltaXY = 0
        if (X == Y):
            deltaXY = 1
        # NB you can make this much more efficient: no function calls etc
        answer = (4*e4*(-12.333333333333334 - 8/(-1 + c) + 11*c + c**2 + c**3/3. + 16*np.log(-1 + c)) + 2*eSq*s*((c**2*gexgey*(MZsq - s))/(gammaZsq*MZsq + MZsqmsSq) + ((1 + c)**2*deltaXY*gexgey)/s + (c*gexgey*(-MZsq + s))/(gammaZsq*MZsq + MZsqmsSq) + (c**3*gexgey*(-MZsq + s))/(3.*(gammaZsq*MZsq + MZsqmsSq)) + (2*c**2*deltaXY*gexgey*(-MZsq + s))/(gammaZsq*MZsq + MZsqmsSq) + (4*(1 + c)*deltaXY*gexgey*(MZsq + s))/s**2 + (16*deltaXY*gammaZ*gexgey*MZ*(MZsq + s)*np.arctan((2*MZsq + s - c*s)/(2.*gammaZ*MZ)))/s**3 + (4*deltaXY*gexgey*(-(gammaZsq*MZsq) + (MZsq + s)**2)*np.log(4*gammaZsq*MZsq + (2*MZsq + s - c*s)**2))/s**3 + c*CXY.real - c**2*CXY.real + (c**3*CXY.real)/3. + 2*c**2*deltaXY*CXY.real + ((1 + c)**3*deltaXY*CXY.real)/3.) + (4*eSq*s*((gexgey*(-2*gammaZsq*MZsq*(4*(2 + (-1 + 2*c + c**2)*deltaXY)*MZsq - (-4 - c*(-4 + deltaXY) + 5*deltaXY + 3*c**2*deltaXY + c**3*deltaXY)*s) - (MZsq - s)*(2*MZsq + s - c*s)*(4*(2 + (-1 + 2*c + c**2)*deltaXY)*MZsq - (8 + (-7 + 3*c + 3*c**2 + c**3)*deltaXY)*s)))/((gammaZsq*MZsq + MZsqmsSq)*(4*gammaZsq*MZsq + (2*MZsq + s - c*s)**2)) + 2*(2 + (-1 + 2*c + c**2)*deltaXY)*CXY.real))/(-1 + c) - (8*gexgey*np.arctan((-2*MZsq - s + c*s)/(2.*gammaZ*MZ))*(deltaXY*gexgey*MZsq*(gammaZsq - MZsq - 2*s) - gexgey*ssq + 2*gammaZ*MZ*(deltaXY*MZsq*(-gammaZsq + MZsq + 2*s) + ssq)*CXY.imag + 4*deltaXY*gammaZsq*MZsq*(MZsq + s)*CXY.real))/(gammaZ*MZ*s) + (8*gexgey*np.log(4*gammaZsq*MZsq + (2*MZsq + s - c*s)**2)*(deltaXY*gexgey*(MZsq + s) - 2*deltaXY*gammaZ*MZ*(MZsq + s)*CXY.imag + (deltaXY*MZsq*(-gammaZsq + MZsq + 2*s) + ssq)*CXY.real))/s + (c**3*deltaXY*ssq*(CXY.imag**2 + CXY.real**2))/3. + (c*(3 + c**2 + c*(-3 + 6*deltaXY))*ssq*(gexgey**2 - 2*gammaZ*gexgey*MZ*CXY.imag + (gammaZsq*MZsq + MZsqmsSq)*CXY.imag**2 + 2*gexgey*(-MZsq + s)*CXY.real + (gammaZsq*MZsq + MZsqmsSq)*CXY.real**2))/(3.*(gammaZsq*MZsq + MZsqmsSq)) + c*(4*deltaXY*gexgey**2 - 8*deltaXY*gammaZ*gexgey*MZ*CXY.imag + (4 - 3*deltaXY)*ssq*CXY.imag**2 + 4*deltaXY*gexgey*(2*MZsq + 3*s)*CXY.real + (4 - 3*deltaXY)*ssq*CXY.real**2) + c**2*deltaXY*s*(s*CXY.imag**2 + CXY.real*(2*gexgey + s*CXY.real)) + (deltaXY**2*s**4*((-16*gexgey*np.log(4*gammaZsq*MZsq + (2*MZsq + s - c*s)**2)*(gexgey*(-2*gammaZsq*MZsq*(5*MZsq - s)*(MZsq + s)**2 + (MZsq - s)*(MZsq + s)**4 + gammaZ4*(5*MZ6 + 3*MZ4*s)) + 4*gammaZ*MZ*(MZsq + s)*(-(gammaZ4*MZ4) + 4*gammaZsq*MZ4*s + (MZ4 - ssq)**2)*CXY.imag - (gammaZ6*MZ6 + MZsqmsSq*(MZsq + s)**4 - gammaZsq*MZsq*(MZsq + s)**2*(5*MZ4 - 14*MZsq*s + 5*ssq) - gammaZ4*(5*MZ8 + 14*MZ6*s + 5*MZ4*ssq))*CXY.real))/s**5 + (c**2*(gexgey**2*(-4*MZ6 - 8*MZ4*s + 11*s**3 + 4*gammaZsq*(3*MZ4 + 2*MZsq*s) + MZsq*ssq) - 2*gammaZ*gexgey*MZ*(4*MZ6 - 2*MZ4*s + 7*s**3 + gammaZsq*(4*MZ4 + 6*MZsq*s) - 8*MZsq*ssq)*CXY.imag + 2*(gammaZsq*MZsq + MZsqmsSq)*s**3*CXY.imag**2 + gexgey*(-4*gammaZ4*MZ4 + 4*MZ8 + 4*MZ6*s - 12*MZsq*s**3 + 13*s**4 + gammaZsq*MZsq*s*(20*MZsq + 7*s) - 9*MZ4*ssq)*CXY.real + 2*(gammaZsq*MZsq + MZsqmsSq)*s**3*CXY.real**2))/s**3 + (c*(-2*gexgey**2*(8*gammaZ4*MZ4 + 8*MZ8 + 20*MZ6*s - 19*MZsq*s**3 - 15*s**4 + 6*MZ4*ssq - 6*gammaZsq*MZsq*(8*MZ4 + 10*MZsq*s + ssq)) + gammaZ*gexgey*MZ*(16*gammaZ4*MZ4 - 48*MZ8 - 16*MZ6*s + 24*MZsq*s**3 - 69*s**4 + 108*MZ4*ssq - 4*gammaZsq*(8*MZ6 + 36*MZ4*s + 13*MZsq*ssq))*CXY.imag + (gammaZsq*MZsq + MZsqmsSq)*s**4*CXY.imag**2 + gexgey*(16*MZ**10 + 24*MZ8*s - 50*MZ4*s**3 + 7*MZsq*s**4 + 31*s**5 - 8*gammaZ4*(6*MZ6 + 7*MZ4*s) - 28*MZ6*ssq + gammaZsq*(-32*MZ8 + 96*MZ6*s - 26*MZsq*s**3 + 132*MZ4*ssq))*CXY.real + (gammaZsq*MZsq + MZsqmsSq)*s**4*CXY.real**2))/s**4 + (2*c**3*(gexgey**2*(-2*MZ4 + 2*gammaZsq*MZsq - 3*MZsq*s + 5*ssq) - gammaZ*gexgey*MZ*(2*MZ4 + 2*gammaZsq*MZsq - 4*MZsq*s + 5*ssq)*CXY.imag + 3*(gammaZsq*MZsq + MZsqmsSq)*ssq*CXY.imag**2 + gexgey*(2*MZ6 + MZ4*s + 8*s**3 + gammaZsq*(2*MZ4 + 5*MZsq*s) - 11*MZsq*ssq)*CXY.real + 3*(gammaZsq*MZsq + MZsqmsSq)*ssq*CXY.real**2))/(3.*s**2) + (c**5*(-(gammaZ*gexgey*MZ*CXY.imag) + (gammaZsq*MZsq + MZsqmsSq)*CXY.imag**2 + CXY.real*(gexgey*(-MZsq + s) + (gammaZsq*MZsq + MZsqmsSq)*CXY.real)))/5. + (c**4*(-2*gammaZ*gexgey*MZ*s*CXY.imag + 2*(gammaZsq*MZsq + MZsqmsSq)*s*CXY.imag**2 + (gexgey*(-MZsq + s) + (gammaZsq*MZsq + MZsqmsSq)*CXY.real)*(gexgey + 2*s*CXY.real)))/(2.*s) - (32*gexgey*np.arctan((-2*MZsq - s + c*s)/(2.*gammaZ*MZ))*((gammaZ6*MZ6 + MZsqmsSq*(MZsq + s)**4 - gammaZsq*MZsq*(MZsq + s)**2*(5*MZ4 - 14*MZsq*s + 5*ssq) - gammaZ4*(5*MZ8 + 14*MZ6*s + 5*MZ4*ssq))*CXY.imag + gammaZ*MZ*(gexgey*(-(gammaZ4*MZ4) - (5*MZsq - 3*s)*(MZsq + s)**3 + 2*gammaZsq*MZsq*(5*MZ4 + 6*MZsq*s + ssq)) + 4*(MZsq + s)*(-(gammaZ4*MZ4) + 4*gammaZsq*MZ4*s + (MZ4 - ssq)**2)*CXY.real)))/s**5))/(gammaZsq*MZsq + MZsqmsSq))/4.
        return answer
    # My ordering for X and Y is (1, 2):=(L, R). 
    for X in range(1, 2):
        for Y in range(1 ,2):
            sigma_tot =+ f(cthmax, X, Y, 1) - f(cthmin, X, Y, 1)
            sigma_tot_SM =+ f(cthmax, X, Y, 0) - f(cthmin, X, Y, 0)
    return sigma_tot / sigma_tot_SM

def ee_ee_obs(wc_obj, par, E, costhmin, costhmax):
    scale = flavio.config['renormalization scale']['ee_ww'] # Use LEP2 renorm scale
    C = wc_obj.get_wcxf(sector='all', scale=scale, par=par,
                        eft='SMEFT', basis='Warsaw')
    return ee_ee(C, par, E, costhmin, costhmax)

_process_tex = r"e^+e^- \to l^+l^-"
_process_taxonomy = r'Process :: $e^+e^-$ scattering :: $e^+e^-\to l^+l^-$ :: $' + _process_tex + r"$"

_obs_name = "R_sigma(ee->ee)"
_obs = Observable(_obs_name)
_obs.arguments = ['E', 'costhmin', 'costhmax']
Prediction(_obs_name, ee_ee_obs)
_obs.set_description(r"Ratio of cross section of $" + _process_tex + r"$ at energy $E$ to that of the SM")
_obs.tex = r"$R_\sigma(" + _process_tex + r")$"
_obs.add_taxonomy(_process_taxonomy)

