r"""Functions for inclusive semi-leptonic $B$ decays.

See arXiv:1107.3100."""


import flavio
from flavio.physics.bdecays.wilsoncoefficients import get_wceff_fccc_std
from math import pi, log, sqrt
from flavio.classes import Observable, Prediction

def BR_BXclnu(par, wc_obj, lep):
    r"""Total branching ratio of $\bar B^0\to X_c \ell^- \bar\nu_\ell$"""
    GF = par['GF']
    scale = flavio.config['renormalization scale']['bxlnu']
    kinetic_cutoff = 1. # cutoff related to the kinetic definition of mb in GeV
    # mb in the kinetic scheme
    mb = flavio.physics.running.running.get_mb_KS(par, kinetic_cutoff)
    # mc in MSbar at 3 GeV
    mc = flavio.physics.running.running.get_mc(par, 3)
    mb_MSbar = flavio.physics.running.running.get_mb(par, scale)
    rho = mc**2/mb**2
    Vcb = flavio.physics.ckm.get_ckm(par)[1, 2]
    alpha_s = flavio.physics.running.running.get_alpha(par, scale, nf_out=5)['alpha_s']
    # wc: NB this includes the EW correction already
    # the b quark mass is MSbar here as it comes from the definition
    # of the scalar operators
    wc = get_wceff_fccc_std(wc_obj, par, 'bc', lep, mb_MSbar, scale, nf=5)
    Gamma_LO = GF**2 * mb**5 / 192. / pi**3 * abs(Vcb)**2 * g(rho)
    r_WC = ( abs(wc['V'])**2 + abs(wc['Vp'])**2
                 - gLR(rho)/g(rho) * (wc['V']*wc['Vp']).real
                 + mb_MSbar**2/4. * ( abs(wc['S'])**2 + abs(wc['Sp'])**2)
                 + mb_MSbar**2/2. * gLR(rho)/g(rho) * (wc['S']*wc['Sp']).real
                 + 12*abs(wc['T'])**2 )
    # eq. (26) of arXiv:1107.3100 + corrections (P. Gambino, private communication)
    r_BLO = ( 1
                 # NLO QCD
                 + alpha_s/pi * pc1(rho, mb)
                 # NNLO QCD
                 + alpha_s**2/pi**2 * pc2(rho, mb)
                 # power correction
                 - par['mu_pi^2']/(2*mb**2)
                 + (1/2. -  2*(1-rho)**4/g(rho))*(par['mu_G^2'] - (par['rho_LS^3'] + par['rho_D^3'])/mb)/mb**2
                 + d(rho)/g(rho) * par['rho_D^3']/mb**3
                 # O(alpha_s) power correction (only numerically)
                 + alpha_s/pi *  par['mu_pi^2'] * 0.071943
                 + alpha_s/pi *  par['mu_G^2'] * (-0.114774)
            )
    # average of B0 and B+ lifetimes
    return (par['tau_B0']+par['tau_B+'])/2. * Gamma_LO * r_WC * r_BLO

def g(rho):
    return 1 - 8*rho + 8*rho**3 - rho**4 - 12*rho**2*log(rho)

def gLR(rho):
    return 4*sqrt(rho)*(1 + 9*rho - 9*rho**2 - rho**3 + 6*rho*(1 + rho)*log(rho))

def d(rho):
    return 8*log(rho) - 10*rho**4/3. + 32*rho**3/3. - 8*rho**2 - 32*rho/3. + 34/3.

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

_process_taxonomy = r'Process :: $b$ hadron decays :: Semi-leptonic tree-level decays :: $B\to X\ell\nu$ :: $'

_tex = {'e': 'e', 'mu': '\mu', 'l': r'\ell'}

for l in ['e', 'mu', 'l']:
        _obs_name = "BR(B->Xc"+l+"nu)"
        _process_tex = r"B\to X_c"+_tex[l]+r"^+\nu_"+_tex[l]
        _obs = Observable(_obs_name)
        _obs.set_description(r"Total branching ratio of $" + _process_tex + r"$")
        _obs.tex = r"$\text{BR}(" + _process_tex + r")$"
        _obs.add_taxonomy(_process_taxonomy + _process_tex + r"$")
        Prediction(_obs_name, BR_tot_function(l))
