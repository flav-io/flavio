r"""Functions for $K\to\ell\nu$ decays."""

import flavio
from flavio.physics.bdecays.blnu import br_plnu_general
from math import pi, log
from flavio.math.functions import li2

def br_klnu(wc_obj, par, lep):
    r"""Branching ratio of $K^+\to\ell^+\nu_\ell$."""
    # CKM element
    Vus = flavio.physics.ckm.get_ckm(par)[0,1]
    # renormalization scale is m_rho
    scale = par['m_rho0']
    # Wilson coefficients
    wc = wc_obj.get_wc('su' + lep + 'nu', scale, par)
    # add SM contribution to Wilson coefficient
    wc['CV_su'+lep+'nu'] += flavio.physics.bdecays.wilsoncoefficients.get_CVSM(par, 'su', scale)
    return br_plnu_general(wc, par, Vus, 'K+', lep, delta=delta_Plnu(par, 'K+', lep))

def r_klnu(wc_obj, par):
    # resumming logs according to (111) of 0707.4464
    # (this is negligibly small for the individual rates)
    rg_corr = 1.00055
    return rg_corr*br_klnu(wc_obj, par, 'e')/br_klnu(wc_obj, par, 'mu')

def delta_Plnu(par, P, lep):
    mrho = par['m_rho0']
    mP = par['m_'+P]
    ml = par['m_'+lep]
    alpha_e = 1/137.035999139 # this is alpha_e(0), a constant for our purposes
    c1 = par['c1_'+P+'lnu'] # e.g. c1_K+lnu
    c2 = par['c2_'+P+'lnu']
    c3 = par['c3_'+P+'lnu']
    if lep=='mu':
        c4 = par['c4_'+P+'munu']
    elif lep=='e':
        c4 = 0 # c_4 tends to zero with vanishing lepton mass
    c2t = par['c2t_'+P+'lnu']
    return alpha_e/pi * (F(ml**2/mP**2) - 3/2.*log(mrho/mP) - c1
            - ml**2/mrho**2 * (c2 * log(mrho**2/ml**2) + c3 + c4)
            + mP**2/mrho**2 * c2t * log(mrho**2/ml**2) )

def F(z):
    return ( 3/2.*log(z) + (13-19*z)/(8*(1-z))
            - (8-5*z)/(4*(1-z)**2)*z*log(z)
            - ((1+z)/(1-z)*log(z)+2)*log(1-z)
            -2*(1+z)/(1-z)*li2(1-z) )

# function returning function needed for prediction instance
def br_klnu_fct(lep):
    def f(wc_obj, par):
        return br_klnu(wc_obj, par, lep)
    return f

# Observable and Prediction instances

# Individual (e and mu) modes
_tex = {'e': 'e', 'mu': '\mu'}
for l in ['e', 'mu']:
    _obs_name = "BR(K+->"+l+"nu)"
    _obs = flavio.classes.Observable(_obs_name)
    _obs.set_description(r"Branching ratio of $K^+\to "+_tex[l]+r"^+\nu_"+_tex[l]+r"(\gamma)$")
    _obs.tex = r"$\text{BR}(K^+\to "+_tex[l]+r"^+\nu_"+_tex[l]+r")$"
    flavio.classes.Prediction(_obs_name, br_klnu_fct(l))

# e/mu ratio
_obs_name = "Remu(K+->lnu)"
_obs = flavio.classes.Observable(_obs_name)
_obs.set_description(r"Ratio of branching ratios of $K^+\to e^+\nu_e$ and $K^+\to \mu^+\nu_\mu$")
_obs.tex = r"$R_{e\mu}(K^+\to \ell^+\nu)$"
flavio.classes.Prediction(_obs_name, r_klnu)
