r"""Functions for inclusive semi-leptonic $B$ decays.

See arXiv:1107.3100."""


import flavio
from flavio.physics.bdecays.wilsoncoefficients import get_wceff_fccc_std
from math import pi
from cmath import log, sqrt
from flavio.classes import Observable, Prediction
from flavio.config import config
from functools import lru_cache


def BR_BXclnu(par, wc_obj, lep):
    r"""Total branching ratio of $\bar B^0\to X_c \ell^- \bar\nu_\ell$"""
    return sum([_BR_BXclnu(par, wc_obj, lep, nu) for nu in ['e','mu','tau']])

def _BR_BXclnu(par, wc_obj, lep, nu):
    GF = par['GF']
    scale = flavio.config['renormalization scale']['bxlnu']
    mb_MSbar = flavio.physics.running.running.get_mb(par, scale)
    wc = get_wceff_fccc_std(wc_obj, par, 'bc', lep, nu, mb_MSbar, scale, nf=5)
    if lep != nu and all(C == 0 for C in wc.values()):
        return 0  # if all WCs vanish, so does the BR!
    kinetic_cutoff = 1. # cutoff related to the kinetic definition of mb in GeV
    # mb in the kinetic scheme
    mb = flavio.physics.running.running.get_mb_KS(par, kinetic_cutoff)
    xl = par['m_'+lep]**2/mb**2
    # mc in MSbar at 3 GeV
    mc = flavio.physics.running.running.get_mc(par, 3)
    xc = mc**2/mb**2
    Vcb = flavio.physics.ckm.get_ckm(par)[1, 2]
    alpha_s = flavio.physics.running.running.get_alpha(par, scale, nf_out=5)['alpha_s']
    # wc: NB this includes the EW correction already
    # the b quark mass is MSbar here as it comes from the definition
    # of the scalar operators
    Gamma_LO = GF**2 * mb**5 / 192. / pi**3 * abs(Vcb)**2
    r_WC = (   g(xc, xl)      * (abs(wc['VL'])**2 + abs(wc['VR'])**2)
             - gLR(xc, xl)    * (wc['VL']*wc['VR']).real
             + g(xc, xl)/4.   * (abs(wc['SR'])**2 + abs(wc['SL'])**2)
             + gLR(xc, xl)/2. * (wc['SR']*wc['SL']).real
             + 12*g(xc, xl)   * abs(wc['T'])**2
             # the following terms vanish for vanishing lepton mass
             + gVS(xc, xl)    * ((wc['VL']*wc['SR']).real
                                       + (wc['VR']*wc['SL']).real)
             + gVSp(xc, xl)   * ((wc['VL']*wc['SL']).real
                                       + (wc['VR']*wc['SR']).real)
             - 12*gVSp(xc, xl)* (wc['VL']*wc['T']).real
             + 12*gVS(xc, xl) * (wc['VR']*wc['T']).real
           )
    # eq. (26) of arXiv:1107.3100 + corrections (P. Gambino, private communication)
    flavio.citations.register("Gambino:2011cq")
    r_BLO = ( 1
                 # NLO QCD
                 + alpha_s/pi * pc1(xc, mb)
                 # NNLO QCD
                 + alpha_s**2/pi**2 * pc2(xc, mb)
                 # power correction
                 - par['mu_pi^2']/(2*mb**2)
                 + (1/2. -  2*(1-xc)**4/g(xc, 0))*(par['mu_G^2'] - (par['rho_LS^3'] + par['rho_D^3'])/mb)/mb**2
                 + d(xc)/g(xc, 0) * par['rho_D^3']/mb**3
                 # O(alpha_s) power correction (only numerically)
                 + alpha_s/pi *  par['mu_pi^2'] * 0.071943
                 + alpha_s/pi *  par['mu_G^2'] * (-0.114774)
            )
    # average of B0 and B+ lifetimes
    r_rem = (1  + par['delta_BXlnu']) # residual pert & non-pert uncertainty
    return (par['tau_B0']+par['tau_B+'])/2. * Gamma_LO * r_WC * r_BLO * r_rem

@lru_cache(maxsize=config['settings']['cache size'])
def g(xc, xl):
    if xl == 0:
        return (1 - 8*xc + 8*xc**3 - xc**4 - 12*xc**2*log(xc)).real
    else:
        return (sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - 7*xc*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - 7*xc**2*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        + xc**3*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - 7*xl*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        + 12*xc*xl*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - 7*xc**2*xl*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - 7*xl**2*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - 7*xc*xl**2*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        + xl**3*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        + 24*xc**2*log(2) - 24*xl**2*log(2) - 24*(-1 + xc**2)*xl**2* log(1 - sqrt(xc))
        + 12*xc**2*log(xc) - 6*xl**2*log(xc) - 6*xc**2*xl**2*log(xc)
        - 12*xl**2*log(sqrt(xc) - 2*xc + xc**1.5)
        + 12*xc**2*xl**2* log(sqrt(xc) - 2*xc + xc**1.5)
        - 12*xl**2*log(xl) + 12*xc**2*xl**2*log(xl)
        - 24*xc**2*log(1 + xc - xl - sqrt(1 + (xc - xl)**2 - 2*(xc + xl)))
        + 12*xl**2* log(1 + xc - xl - sqrt(1 + (xc - xl)**2 - 2*(xc + xl)))
        + 12*xc**2*xl**2* log(1 + xc - xl - sqrt(1 + (xc - xl)**2 - 2*(xc + xl)))
        + 12*xl**2* log(1 + xc**2 - xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - xc*(2 + xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))))
        - 12*xc**2*xl**2* log(1 + xc**2 - xl
        + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - xc*(2 + xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))))).real

@lru_cache(maxsize=config['settings']['cache size'])
def gLR(xc, xl):
    if xl == 0:
        return (4*sqrt(xc)*(1 + 9*xc - 9*xc**2 - xc**3 + 6*xc*(1 + xc)*log(xc))).real
    else:
        return (4*sqrt(xc)*(sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        + 10*xc*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        + xc**2*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - 5*xl*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - 5*xc*xl*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - 2*xl**2*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - 12*xc*log(2) - 12*xc**2*log(2) + 24*xc*xl*log(2)
        - 12*xl**2*log(2) - 12*(-1 + xc)*xl**2* log(1 - sqrt(xc))
        - 6*xc*log(xc) - 6*xc**2*log(xc) + 12*xc*xl*log(xc) - 3*xl**2*log(xc)
        - 3*xc*xl**2*log(xc) - 6*xl**2*log(sqrt(xc) - 2*xc + xc**1.5)
        + 6*xc*xl**2* log(sqrt(xc) - 2*xc + xc**1.5) - 6*xl**2*log(xl)
        + 6*xc*xl**2*log(xl) + 12*xc*log(1 + xc - xl
        - sqrt(1 + (xc - xl)**2 - 2*(xc + xl)))
        + 12*xc**2*log(1 + xc - xl - sqrt(1 + (xc - xl)**2 - 2*(xc + xl)))
        - 24*xc*xl*log(1 + xc - xl - sqrt(1 + (xc - xl)**2 - 2*(xc + xl)))
        + 6*xl**2*log(1 + xc - xl - sqrt(1 + (xc - xl)**2 - 2*(xc + xl)))
        + 6*xc*xl**2* log(1 + xc - xl - sqrt(1 + (xc - xl)**2 - 2*(xc + xl)))
        + 6*xl**2*log(1 + xc**2 - xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - xc*(2 + xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))))
        - 6*xc*xl**2* log(1 + xc**2 - xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - xc*(2 + xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl)))))).real

@lru_cache(maxsize=config['settings']['cache size'])
def gVS(xc, xl):
    if xl == 0:
        return 0
    else:
        return (2*sqrt(xl)*(sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - 5*xc*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - 2*xc**2*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        + 10*xl*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - 5*xc*xl*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        + xl**2*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl)) + 12*xc**2*log(2)
        + 12*xl*log(2) - 24*xc*xl*log(2) + 12*xl**2*log(2)
        - 12*xl*(1 - 2*xc + xc**2 + xl)* log(1 - sqrt(xc)) + 6*xc**2*log(xc)
        + 3*xl*log(xc) - 6*xc*xl*log(xc) - 3*xc**2*xl*log(xc) + 3*xl**2*log(xc)
        + 6*xl*log(sqrt(xc) - 2*xc + xc**1.5) - 12*xc*xl*log(sqrt(xc) - 2*xc + xc**1.5)
        + 6*xc**2*xl* log(sqrt(xc) - 2*xc + xc**1.5)
        + 6*xl**2*log(sqrt(xc) - 2*xc + xc**1.5) + 6*xl*log(xl)
        - 12*xc*xl*log(xl) + 6*xc**2*xl*log(xl) + 6*xl**2*log(xl)
        - 12*xc**2*log(1 + xc - xl - sqrt(1 + (xc - xl)**2 - 2*(xc + xl)))
        - 6*xl*log(1 + xc - xl - sqrt(1 + (xc - xl)**2 - 2*(xc + xl)))
        + 12*xc*xl*log(1 + xc - xl - sqrt(1 + (xc - xl)**2 - 2*(xc + xl)))
        + 6*xc**2*xl* log(1 + xc - xl - sqrt(1 + (xc - xl)**2 - 2*(xc + xl)))
        - 6*xl**2*log(1 + xc - xl - sqrt(1 + (xc - xl)**2 - 2*(xc + xl)))
        - 6*xl*log(1 + xc**2 - xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - xc*(2 + xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))))
        + 12*xc*xl*log(1 + xc**2 - xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - xc*(2 + xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))))
        - 6*xc**2*xl* log(1 + xc**2 - xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - xc*(2 + xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))))
        - 6*xl**2*log(1 + xc**2 - xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - xc*(2 + xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl)))))).real

@lru_cache(maxsize=config['settings']['cache size'])
def gVSp(xc, xl):
    if xl == 0:
        return 0
    else:
        return (2*sqrt(xc*xl)* (2*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        + 5*xc*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - xc**2*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        + 5*xl*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - 10*xc*xl*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - xl**2*sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl)) - 12*xc*log(2)
        + 12*xl*log(2) - 12*(1 + xc**2 + xc*(-2 + xl))*xl* log(1 - sqrt(xc))
        - 6*xc*log(xc) + 3*xl*log(xc) + 6*xc*xl*log(xc) - 3*xc**2*xl*log(xc)
        - 3*xc*xl**2*log(xc) + 6*xl*log(sqrt(xc) - 2*xc + xc**1.5)
        - 12*xc*xl*log(sqrt(xc) - 2*xc + xc**1.5)
        + 6*xc**2*xl* log(sqrt(xc) - 2*xc + xc**1.5)
        + 6*xc*xl**2* log(sqrt(xc) - 2*xc + xc**1.5)
        + 6*xl*log(xl) - 12*xc*xl*log(xl) + 6*xc**2*xl*log(xl) + 6*xc*xl**2*log(xl)
        + 12*xc*log(1 + xc - xl - sqrt(1 + (xc - xl)**2 - 2*(xc + xl)))
        - 6*xl*log(1 + xc - xl - sqrt(1 + (xc - xl)**2 - 2*(xc + xl)))
        - 12*xc*xl*log(1 + xc - xl - sqrt(1 + (xc - xl)**2 - 2*(xc + xl)))
        + 6*xc**2*xl* log(1 + xc - xl - sqrt(1 + (xc - xl)**2 - 2*(xc + xl)))
        + 6*xc*xl**2* log(1 + xc - xl - sqrt(1 + (xc - xl)**2 - 2*(xc + xl)))
        - 6*xl*log(1 + xc**2 - xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - xc*(2 + xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))))
        + 12*xc*xl*log(1 + xc**2 - xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - xc*(2 + xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))))
        - 6*xc**2*xl* log(1 + xc**2 - xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - xc*(2 + xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))))
        - 6*xc*xl**2* log(1 + xc**2 - xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl))
        - xc*(2 + xl + sqrt(xc**2 + (-1 + xl)**2 - 2*xc*(1 + xl)))))).real

def d(xc):
    return (8*log(xc) - 10*xc**4/3. + 32*xc**3/3. - 8*xc**2 - 32*xc/3. + 34/3.).real

def pc1(r, mb):
    # this is an expansion to 2nd order in mb around 4.6 and in r around 0.05
    # P. Gambino,  private communication
    # kinetic scheme cutoff is set to 1 GeV
    return ( 6.486085393242938 - 80.16227770322831*r + 207.37836204469366*r**2
            + mb*(-2.3090743981240274 + 14.029509187000471*r - 36.61694487623083*r**2)
            + mb**2*(0.18126017716432158 - 0.8813205571033417*r + 3.1906139935867635*r**2))

def pc2(r, mb):
    # this is an expansion to 2nd order in mb around 4.6 and in r around 0.05
    # P. Gambino,  private communication
    # kinetic scheme cutoff is set to 1 GeV
    return  ( 63.344451026174276 - 1060.9791881246733*r + 4332.058337615373*r**2
             + mb*(-21.760717863346223 + 273.7460360545832*r - 1032.068345746423*r**2)
             + mb**2*(1.8406501267881998 - 20.26973707297946*r + 73.82649433414315*r**2))


def BR_tot_function(lep):
    if lep == 'l':
        return lambda wc_obj, par: (BR_BXclnu(par, wc_obj, 'e')+BR_BXclnu(par, wc_obj, 'mu'))/2
    else:
        return lambda wc_obj, par: BR_BXclnu(par, wc_obj, lep)

def BR_tot_leptonflavour(wc_obj, par, lnum, lden):
    if lnum == 'l':
        num = (BR_BXclnu(par, wc_obj, 'e') + BR_BXclnu(par, wc_obj, 'mu'))/2.
    else:
        num = BR_BXclnu(par, wc_obj, lnum)
    if num == 0:
        return 0
    if lden == 'l':
        den = (BR_BXclnu(par, wc_obj, 'e') + BR_BXclnu(par, wc_obj, 'mu'))/2.
    else:
        den = BR_BXclnu(par, wc_obj, lden)
    return num/den

def BR_tot_leptonflavour_function(lnum, lden):
    return lambda wc_obj, par: BR_tot_leptonflavour(wc_obj, par, lnum, lden)

_process_taxonomy = r'Process :: $b$ hadron decays :: Semi-leptonic tree-level decays :: $B\to X\ell\nu$ :: $'

_lep = {'e': 'e', 'mu': r'\mu', 'tau': r'\tau', 'l': r'\ell'}

for l in _lep:
        _obs_name = "BR(B->Xc"+l+"nu)"
        _process_tex = r"B\to X_c"+_lep[l]+r"^+\nu_"+_lep[l]
        _obs = Observable(_obs_name)
        _obs.set_description(r"Total branching ratio of $" + _process_tex + r"$")
        _obs.tex = r"$\text{BR}(" + _process_tex + r")$"
        _obs.add_taxonomy(_process_taxonomy + _process_tex + r"$")
        Prediction(_obs_name, BR_tot_function(l))

# Lepton flavour ratios
for l in [('mu','e'), ('tau','mu'), ('tau', 'l')]:
    _obs_name = "R"+l[0]+l[1]+"(B->Xclnu)"
    _obs = Observable(name=_obs_name)
    _process_1 = r"B\to X_c"+_lep[l[0]]+r"^+\nu_"+_lep[l[0]]
    _process_2 = r"B\to X_c"+_lep[l[1]]+r"^+\nu_"+_lep[l[1]]
    _obs.set_description(r"Ratio of total branching ratios of $" + _process_1 + r"$" + " and " + r"$" + _process_2 +r"$")
    _obs.tex = r"$R_{" + _lep[l[0]] + ' ' + _lep[l[1]] + r"}(B\to X_c\ell^+\nu)$"
        # add taxonomy for both processes (e.g. B->Xcenu and B->Xcmunu)
    _obs.add_taxonomy(_process_taxonomy + _process_1 + r"$")
    _obs.add_taxonomy(_process_taxonomy + _process_2 + r"$")
    Prediction(_obs_name, BR_tot_leptonflavour_function(l[0], l[1]))
