r"""Functions for $K\to\ell\nu$ and $\pi\to\ell\nu$ decays."""

import flavio
from flavio.physics.bdecays.blnu import br_plnu_general
from math import pi, log
from flavio.math.functions import li2

def br_plnu(wc_obj, par, P, lep):
    r"""Branching ratio of $P^+\to\ell^+\nu_\ell$."""
    # CKM element
    if P=='K+':
        Vij = flavio.physics.ckm.get_ckm(par)[0,1]
        qiqj = 'su'
    elif P=='pi+':
        Vij = flavio.physics.ckm.get_ckm(par)[0,0]
        qiqj = 'du'
    # renormalization scale is m_rho
    scale = par['m_rho0']
    # Wilson coefficients
    wc = wc_obj.get_wc(qiqj + lep + 'nu', scale, par)
    # add SM contribution to Wilson coefficient
    wc['CV_'+qiqj+lep+'nu'] += flavio.physics.bdecays.wilsoncoefficients.get_CVSM(par, scale)
    return br_plnu_general(wc, par, Vij, P, lep, delta=delta_Plnu(par, P, lep))

def r_plnu(wc_obj, par, P):
    # resumming logs according to (111) of 0707.4464
    # (this is negligibly small for the individual rates)
    rg_corr = 1.00055
    return rg_corr*br_plnu(wc_obj, par, P, 'e')/br_plnu(wc_obj, par, P, 'mu')

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
def br_plnu_fct(P, lep):
    def f(wc_obj, par):
        return br_plnu(wc_obj, par, P, lep)
    return f

def r_plnu_fct(P):
    def f(wc_obj, par):
        return r_plnu(wc_obj, par, P)
    return f

# Observable and Prediction instances

# Individual (e and mu) modes
_tex = {'e': 'e', 'mu': '\mu'}
_tex_p = {'K+': r'K^+', 'pi+': r'\pi^+',}
for P in ['K+', 'pi+']:
    for l in ['e', 'mu']:
        _obs_name = "BR("+P+"->"+l+"nu)"
        _obs = flavio.classes.Observable(_obs_name)
        _obs.set_description(r"Branching ratio of $"+_tex_p[P]+r"\to "+_tex[l]+r"^+\nu_"+_tex[l]+r"(\gamma)$")
        _obs.tex = r"$\text{BR}("+_tex_p[P]+r"\to "+_tex[l]+r"^+\nu_"+_tex[l]+r")$"
        flavio.classes.Prediction(_obs_name, br_plnu_fct(P, l))

    # e/mu ratios
    _obs_name = "Remu("+P+"->lnu)"
    _obs = flavio.classes.Observable(_obs_name)
    _obs.set_description(r"Ratio of branching ratios of $"+_tex_p[P]+r"\to e^+\nu_e$ and $"+_tex_p[P]+r"\to \mu^+\nu_\mu$")
    _obs.tex = r"$R_{e\mu}("+_tex_p[P]+r"\to \ell^+\nu)$"
    flavio.classes.Prediction(_obs_name, r_plnu_fct(P))
