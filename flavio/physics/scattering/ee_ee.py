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
import flavio
from IPython.display import display

# Kronecker delta
def delta(a, b):
    return int(a == b)

# predicted ratio to SM with SMEFT operators of e+e-->mumu for LEP2 energy E and family fam total cross-section. afb should be true for forward-backward asymmetry, whereas it should be false for the total cross-section. cthmin and chtmax are the minimum and maximum values of cosine of the scattering angle in the current bin. Programmed by BCA 24/4/23
def ee_ee(C, par, E, cthmin, cthmax):
    # Check energy E is correct
    if (E != 136.3 and E != 161.3 and E != 172.1 and E != 182.7 and E != 188.6 and E != 191.6 and E != 195.5 and E != 199.5 and E!= 201.8 and E!= 204.8 and E!= 206.5):
        raise ValueError('ee_ee called with incorrect LEP2 energy {} GeV.'.format(E))
    if ((cthmin, cthmax) != (-0.90, -0.72) and
        (cthmin, cthmax) != (-0.72, -0.54) and
        (cthmin, cthmax) != (-0.54, -0.36) and
        (cthmin, cthmax) != (-0.36, -0.18) and
        (cthmin, cthmax) != (-0.18,  -0.0) and
        (cthmin, cthmax) != (  0.0,  0.09) and
        (cthmin, cthmax) != ( 0.09,  0.18) and        
        (cthmin, cthmax) != ( 0.18,  0.27) and
        (cthmin, cthmax) != ( 0.27,  0.36) and
        (cthmin, cthmax) != ( 0.36,  0.45) and
        (cthmin, cthmax) != ( 0.45,  0.54) and
        (cthmin, cthmax) != ( 0.54,  0.63) and
        (cthmin, cthmax) != ( 0.63,  0.72) and
        (cthmin, cthmax) != ( 0.72,  0.81) and
        (cthmin, cthmax) != ( 0.81,  0.90)):
        raise ValueError(f'ee_ee called with incorrect (cthmin={cthmin}, cthmax={cthmax})')
    s = E * E
    ssq = s**2
    scub = ssq * s
    MZ = par['m_Z']
    MZsq = MZ**2
    MZ4 = MZsq**2
    MZ6 = MZ4 * MZsq
    MZ8 = MZ4**2
    MZsqmsSq = (MZsq - s)**2
    gammaZ = 1 / par['tau_Z']
    gammaZsq = gammaZ**2
    gammaZ4 = gammaZsq**2
    gammaZ6 = gammaZ4 * gammaZsq
    gammaZsqMZsq = gammaZsq * MZsq
    GF = par['GF']
    alpha = par['alpha_e']
    s2w   = par['s2w']
    gzeL  = -0.5 + s2w
    gzeR  = s2w
    eSq   = 4 * pi * alpha
    e4    = eSq**2
    gLsq  = eSq / s2w
    gYsq  = gLsq * s2w / (1. - s2w)
    g_cw  = sqrt(gLsq + gYsq)
    gVe  = g_cw * gV_SM('e', par) 
    gAe  = g_cw * gA_SM('e', par)
    # f is proportional to d sigma / d (c=cos theta) integrated over c.
    # c is cos(theta)
    # X is 1 or 2, so is Y (meaning L or R).
    # Z is 1 for including new physics, 0 for SM
    def f(c1, c2, X, Y, Z):
        # SM versions 
        gex = (gVe + gAe) * delta(X, 2) + (gVe - gAe) * delta(X, 1)
        gey = (gVe + gAe) * delta(Y, 2) + (gVe - gAe) * delta(Y, 1)
        CXY = 0
        CYX = 0
        if (Z == 1): # add SMEFT contributions to Z gauge couplings to electrons
            gex += (smeftew.d_gVl('e', 'e', par, C) + smeftew.d_gAl('e', 'e', par, C)) * delta(X, 2) + (smeftew.d_gVl('e', 'e', par, C) - smeftew.d_gAl('e', 'e', par, C)) * delta(X, 1)
            gey += (smeftew.d_gVl('e', 'e', par, C) + smeftew.d_gAl('e', 'e', par, C)) * delta(Y, 2) + (smeftew.d_gVl('e', 'e', par, C) - smeftew.d_gAl('e', 'e', par, C)) * delta(Y, 1)
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
                raise ValueError(f'f called with incorrect X={X},Y={Y}'.format(X,Y))
        gexgey = gex * gey
        deltaXY = 0
        if (X == Y):
            deltaXY = 1
        # NB you can make this much more efficient: no function calls etc
        reCXY = np.real(CXY)
        imCXY = np.imag(CXY)
        # integrated Bhabha scattering cross-section from cos theta=c1 to c2. Obtained with nnI2.nb mathematica noteboook in anc subdirectory
        answer = e4*(8/(-1 + c1) - 11*c1 - c1**2 - c1**3/3. - 8/(-1 + c2) + 11*c2 + c2**2 + c2**3/3. + 16*np.log((1 - c2)/(1 - c1))) - (eSq*s*(-0.3333333333333333*(c1**3*gexgey*(MZsq - s))/(gammaZsqMZsq + MZsqmsSq) + c1*gexgey*((-MZsq + s)/(gammaZsqMZsq + MZsqmsSq) + (deltaXY*(4*MZsq + 6*s))/s**2) + (c1**2*gexgey*((MZsq - s)*s + deltaXY*(MZ4 + gammaZsqMZsq - 4*MZsq*s + 3*ssq)))/((gammaZsqMZsq + MZsqmsSq)*s) + (16*deltaXY*gammaZ*gexgey*MZ*(MZsq + s)*np.arctan((2*MZsq + s - c1*s)/(2.*gammaZ*MZ)))/scub + (4*deltaXY*gexgey*(-(gammaZsqMZsq) + (MZsq + s)**2)*np.log(4*gammaZsqMZsq + (2*MZsq + s - c1*s)**2))/scub + c1*(1 + deltaXY)*reCXY + (c1**3*(1 + deltaXY)*reCXY)/3. + c1**2*(-1 + 3*deltaXY)*reCXY))/2. + (eSq*s*(-0.3333333333333333*(c2**3*gexgey*(MZsq - s))/(gammaZsqMZsq + MZsqmsSq) + c2*gexgey*((-MZsq + s)/(gammaZsqMZsq + MZsqmsSq) + (deltaXY*(4*MZsq + 6*s))/s**2) + (c2**2*gexgey*((MZsq - s)*s + deltaXY*(MZ4 + gammaZsqMZsq - 4*MZsq*s + 3*ssq)))/((gammaZsqMZsq + MZsqmsSq)*s) + (16*deltaXY*gammaZ*gexgey*MZ*(MZsq + s)*np.arctan((2*MZsq + s - c2*s)/(2.*gammaZ*MZ)))/scub + (4*deltaXY*gexgey*(-(gammaZsqMZsq) + (MZsq + s)**2)*np.log(4*gammaZsqMZsq + (2*MZsq + s - c2*s)**2))/scub + c2*(1 + deltaXY)*reCXY + (c2**3*(1 + deltaXY)*reCXY)/3. + c2**2*(-1 + 3*deltaXY)*reCXY))/2. - ((c1 + c1**3/3. + c1**2*(-1 + 2*deltaXY))*ssq*(gexgey**2 - 2*gammaZ*gexgey*MZ*imCXY + (gammaZsqMZsq + MZsqmsSq)*imCXY**2 + 2*gexgey*(-MZsq + s)*reCXY + (gammaZsqMZsq + MZsqmsSq)*reCXY**2))/(4.*(gammaZsqMZsq + MZsqmsSq)) + ((c2 + c2**3/3. + c2**2*(-1 + 2*deltaXY))*ssq*(gexgey**2 - 2*gammaZ*gexgey*MZ*imCXY + (gammaZsqMZsq + MZsqmsSq)*imCXY**2 + 2*gexgey*(-MZsq + s)*reCXY + (gammaZsqMZsq + MZsqmsSq)*reCXY**2))/(4.*(gammaZsqMZsq + MZsqmsSq)) + (eSq*s*((-8*gammaZ*gexgey*(deltaXY*MZsq*(gammaZsq + MZsq) - ssq)*np.arctan((2*MZsq + s - c1*s)/(2.*gammaZ*MZ)))/(MZ*(gammaZsq + MZsq)*s**2) + (8*gammaZ*gexgey*(deltaXY*MZsq*(gammaZsq + MZsq) - ssq)*np.arctan((2*MZsq + s - c2*s)/(2.*gammaZ*MZ)))/(MZ*(gammaZsq + MZsq)*s**2) - (4*gexgey*(deltaXY*(gammaZsq + MZsq)*(MZsq + 2*s) + ssq)*np.log(4*gammaZsqMZsq + (2*MZsq + s - c1*s)**2))/((gammaZsq + MZsq)*s**2) + (4*gexgey*(deltaXY*(gammaZsq + MZsq)*(MZsq + 2*s) + ssq)*np.log(4*gammaZsqMZsq + (2*MZsq + s - c2*s)**2))/((gammaZsq + MZsq)*s**2) - (-1 + c1)**2*deltaXY*((gexgey*(-MZsq + s))/(gammaZsqMZsq + MZsqmsSq) + 2*reCXY) + (-1 + c2)**2*deltaXY*((gexgey*(-MZsq + s))/(gammaZsqMZsq + MZsqmsSq) + 2*reCXY) + 8*np.log((1 - c2)/(1 - c1))*(-((gexgey*((MZsq - s)*((1 + deltaXY)*MZsq - s) + gammaZsq*((1 + deltaXY)*MZsq - deltaXY*s)))/((gammaZsq + MZsq)*(gammaZsqMZsq + MZsqmsSq))) + (1 + deltaXY)*reCXY) - (4*(-1 + c1)*deltaXY*(gexgey*(MZ4 + gammaZsqMZsq - 4*MZsq*s + 3*ssq) + 4*(gammaZsqMZsq + MZsqmsSq)*s*reCXY))/((gammaZsqMZsq + MZsqmsSq)*s) + (4*(-1 + c2)*deltaXY*(gexgey*(MZ4 + gammaZsqMZsq - 4*MZsq*s + 3*ssq) + 4*(gammaZsqMZsq + MZsqmsSq)*s*reCXY))/((gammaZsqMZsq + MZsqmsSq)*s)))/2. - (ssq*((-8*gexgey*np.arctan((-2*MZsq - s + c1*s)/(2.*gammaZ*MZ))*(deltaXY*gexgey*MZsq*(gammaZsq - MZsq - 2*s) - gexgey*ssq + 2*gammaZ*MZ*(deltaXY*MZsq*(-gammaZsq + MZsq + 2*s) + ssq)*imCXY + 4*deltaXY*gammaZsqMZsq*(MZsq + s)*reCXY))/(gammaZ*MZ*scub) + (8*gexgey*np.log(4*gammaZsqMZsq + (2*MZsq + s - c1*s)**2)*(deltaXY*gexgey*(MZsq + s) - 2*deltaXY*gammaZ*MZ*(MZsq + s)*imCXY + (deltaXY*MZsq*(-gammaZsq + MZsq + 2*s) + ssq)*reCXY))/scub + (c1**3*deltaXY*(imCXY**2 + reCXY**2))/3. + (c1*(4*deltaXY*gexgey**2 - 8*deltaXY*gammaZ*gexgey*MZ*imCXY + (4 - 3*deltaXY)*ssq*imCXY**2 + 4*deltaXY*gexgey*(2*MZsq + 3*s)*reCXY + (4 - 3*deltaXY)*ssq*reCXY**2))/s**2 + (c1**2*deltaXY*(s*imCXY**2 + reCXY*(2*gexgey + s*reCXY)))/s))/4. + (ssq*((-8*gexgey*np.arctan((-2*MZsq - s + c2*s)/(2.*gammaZ*MZ))*(deltaXY*gexgey*MZsq*(gammaZsq - MZsq - 2*s) - gexgey*ssq + 2*gammaZ*MZ*(deltaXY*MZsq*(-gammaZsq + MZsq + 2*s) + ssq)*imCXY + 4*deltaXY*gammaZsqMZsq*(MZsq + s)*reCXY))/(gammaZ*MZ*scub) + (8*gexgey*np.log(4*gammaZsqMZsq + (2*MZsq + s - c2*s)**2)*(deltaXY*gexgey*(MZsq + s) - 2*deltaXY*gammaZ*MZ*(MZsq + s)*imCXY + (deltaXY*MZsq*(-gammaZsq + MZsq + 2*s) + ssq)*reCXY))/scub + (c2**3*deltaXY*(imCXY**2 + reCXY**2))/3. + (c2*(4*deltaXY*gexgey**2 - 8*deltaXY*gammaZ*gexgey*MZ*imCXY + (4 - 3*deltaXY)*ssq*imCXY**2 + 4*deltaXY*gexgey*(2*MZsq + 3*s)*reCXY + (4 - 3*deltaXY)*ssq*reCXY**2))/s**2 + (c2**2*deltaXY*(s*imCXY**2 + reCXY*(2*gexgey + s*reCXY)))/s))/4. - (deltaXY*ssq*((4*gexgey*np.log(4*gammaZsqMZsq + (2*MZsq + s - c1*s)**2)*(gexgey*(-((MZsq - s)*(MZsq + s)**2) + gammaZsqMZsq*(3*MZsq + s)) - 2*gammaZ*MZ*(gammaZsqMZsq + MZsqmsSq)*(MZsq + s)*imCXY + (-(gammaZ4*MZ4) + 4*gammaZsq*MZ4*s + (MZ4 - ssq)**2)*reCXY))/scub + (c1*(2*gexgey**2*(-2*MZ4 + 2*gammaZsqMZsq - MZsq*s + 3*ssq) - gammaZ*gexgey*MZ*(4*MZ4 + 4*gammaZsqMZsq - 8*MZsq*s + 5*ssq)*imCXY + (gammaZsqMZsq + MZsqmsSq)*ssq*imCXY**2 + gexgey*(4*MZ6 - 2*MZ4*s + 7*scub + gammaZsq*(4*MZ4 + 6*MZsq*s) - 9*MZsq*ssq)*reCXY + (gammaZsqMZsq + MZsqmsSq)*ssq*reCXY**2))/s**2 + (c1**3*(-(gammaZ*gexgey*MZ*imCXY) + (gammaZsqMZsq + MZsqmsSq)*imCXY**2 + reCXY*(gexgey*(-MZsq + s) + (gammaZsqMZsq + MZsqmsSq)*reCXY)))/3. + (c1**2*(-(gammaZ*gexgey*MZ*s*imCXY) + (gammaZsqMZsq + MZsqmsSq)*s*imCXY**2 + (gexgey*(-MZsq + s) + (gammaZsqMZsq + MZsqmsSq)*reCXY)*(gexgey + s*reCXY)))/s - (8*gexgey*np.arctan((-2*MZsq - s + c1*s)/(2.*gammaZ*MZ))*((-(gammaZ4*MZ4) + 4*gammaZsq*MZ4*s + (MZ4 - ssq)**2)*imCXY + gammaZ*MZ*(gexgey*(-3*MZ4 + gammaZsqMZsq - 2*MZsq*s + ssq) + 2*(gammaZsqMZsq + MZsqmsSq)*(MZsq + s)*reCXY)))/scub))/(2.*(gammaZsqMZsq + MZsqmsSq)) + (deltaXY*ssq*((4*gexgey*np.log(4*gammaZsqMZsq + (2*MZsq + s - c2*s)**2)*(gexgey*(-((MZsq - s)*(MZsq + s)**2) + gammaZsqMZsq*(3*MZsq + s)) - 2*gammaZ*MZ*(gammaZsqMZsq + MZsqmsSq)*(MZsq + s)*imCXY + (-(gammaZ4*MZ4) + 4*gammaZsq*MZ4*s + (MZ4 - ssq)**2)*reCXY))/scub + (c2*(2*gexgey**2*(-2*MZ4 + 2*gammaZsqMZsq - MZsq*s + 3*ssq) - gammaZ*gexgey*MZ*(4*MZ4 + 4*gammaZsqMZsq - 8*MZsq*s + 5*ssq)*imCXY + (gammaZsqMZsq + MZsqmsSq)*ssq*imCXY**2 + gexgey*(4*MZ6 - 2*MZ4*s + 7*scub + gammaZsq*(4*MZ4 + 6*MZsq*s) - 9*MZsq*ssq)*reCXY + (gammaZsqMZsq + MZsqmsSq)*ssq*reCXY**2))/s**2 + (c2**3*(-(gammaZ*gexgey*MZ*imCXY) + (gammaZsqMZsq + MZsqmsSq)*imCXY**2 + reCXY*(gexgey*(-MZsq + s) + (gammaZsqMZsq + MZsqmsSq)*reCXY)))/3. + (c2**2*(-(gammaZ*gexgey*MZ*s*imCXY) + (gammaZsqMZsq + MZsqmsSq)*s*imCXY**2 + (gexgey*(-MZsq + s) + (gammaZsqMZsq + MZsqmsSq)*reCXY)*(gexgey + s*reCXY)))/s - (8*gexgey*np.arctan((-2*MZsq - s + c2*s)/(2.*gammaZ*MZ))*((-(gammaZ4*MZ4) + 4*gammaZsq*MZ4*s + (MZ4 - ssq)**2)*imCXY + gammaZ*MZ*(gexgey*(-3*MZ4 + gammaZsqMZsq - 2*MZsq*s + ssq) + 2*(gammaZsqMZsq + MZsqmsSq)*(MZsq + s)*reCXY)))/scub))/(2.*(gammaZsqMZsq + MZsqmsSq))
        return answer
    sigma_tot    = 0
    sigma_tot_SM = 0
    # My ordering for X and Y is (1, 2):=(L, R). 
    for X in range(1, 2):
        for Y in range(1 ,2):
            sigma_tot += f(cthmin, cthmax, X, Y, 1)
            sigma_tot_SM += f(cthmin, cthmax, X, Y, 0)
    print("DEBUG 1: ",sigma_tot / sigma_tot_SM)
    return sigma_tot / sigma_tot_SM

def ee_ee_obs(wc_obj, par, E, cthmin, cthmax):
    scale = flavio.config['renormalization scale']['ee_ww'] # Use LEP2 renorm scale
    C = wc_obj.get_wcxf(sector='all', scale=scale, par=par,
                        eft='SMEFT', basis='Warsaw')
    return ee_ee(C, par, E, cthmin, cthmax)

_process_tex = r"e^+e^- \to e^+e^-"
_process_taxonomy = r'Process :: $e^+e^-$ scattering :: $e^+e^-\to e^+e^-$ :: $' + _process_tex + r"$"

_obs_name = "R_sigma(ee->ee)"
_obs = Observable(_obs_name)
_obs.arguments = ['E', 'cthmin', 'cthmax']
Prediction(_obs_name, ee_ee_obs)
_obs.set_description(r"Ratio of cross section of $" + _process_tex + r"$ at energy $E$ to that of the SM")
_obs.tex = r"$R_\sigma(" + _process_tex + r")$"
_obs.add_taxonomy(_process_taxonomy)

