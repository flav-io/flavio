r"""Functions for $K\to\ell\nu$ and $\pi\to\ell\nu$ decays."""

import flavio
from flavio.physics.bdecays.blnu import br_plnu_general
from math import pi, log
from flavio.math.functions import li2

def br_klnu(wc_obj, par, P, lep):
    r"""Branching ratio of $P^+\to\ell^+\nu_\ell$."""
    return sum([ _br_klnu(wc_obj,par,P,lep,nu) for nu in ['e','mu','tau']])

def _br_klnu(wc_obj, par, P, lep, nu):
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
    wc = wc_obj.get_wc(qiqj + lep + 'nu' + nu, scale, par, nf_out=3)
    # add SM contribution to Wilson coefficient
    if lep == nu:
        wc['CVL_'+qiqj+lep+'nu'+nu] += flavio.physics.bdecays.wilsoncoefficients.get_CVLSM(par, scale, nf=3)
    ms = flavio.physics.running.running.get_ms(par, scale)
    mu = flavio.physics.running.running.get_mu(par, scale)
    return br_plnu_general(wc, par, Vij, P, qiqj, lep, nu, ms, mu, delta=delta_Plnu(par, P, lep))

def r_klnu(wc_obj, par, P):
    # resumming logs according to (111) of 0707.4464
    # (this is negligibly small for the individual rates)
    rg_corr = 1.00055
    return rg_corr*br_klnu(wc_obj, par, P, 'e')/br_klnu(wc_obj, par, P, 'mu')

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
def br_klnu_fct(P, lep):
    def f(wc_obj, par):
        return br_klnu(wc_obj, par, P, lep)
    return f

def r_klnu_fct(P):
    def f(wc_obj, par):
        return r_klnu(wc_obj, par, P)
    return f

# Observable and Prediction instances

_tex = {'e': 'e', 'mu': '\mu'}
_tex_p = {'K+': r'K^+', 'pi+': r'\pi^+',}

# K+->lnu
for l in ['e', 'mu']:
    # Individual (e and mu) modes
    _obs_name = "BR(K+->"+l+"nu)"
    _obs = flavio.classes.Observable(_obs_name)
    _process_tex = r"K^+\to "+_tex[l]+r"^+\nu_"+_tex[l]
    _process_taxonomy = r'Process :: $s$ hadron decays :: Leptonic tree-level decays :: $K\to \ell\nu$ :: $' + _process_tex + r'$'
    _obs.add_taxonomy(_process_taxonomy)
    _obs.set_description(r"Branching ratio of $" + _process_tex +r"(\gamma)$")
    _obs.tex = r"$\text{BR}(" + _process_tex + r")$"
    flavio.classes.Prediction(_obs_name, br_klnu_fct('K+', l))

# e/mu ratios
_obs_name = "Remu(K+->lnu)"
_obs = flavio.classes.Observable(_obs_name)
_obs.set_description(r"Ratio of branching ratios of $K^+\to e^+\nu_e$ and $K^+\to \mu^+\nu_\mu$")
_obs.tex = r"$R_{e\mu}(K^+\to \ell^+\nu)$"
_obs.add_taxonomy(r'Process :: $s$ hadron decays :: Leptonic tree-level decays :: $K\to \ell\nu$ :: $K^+\to e^+\nu_e$')
_obs.add_taxonomy(r'Process :: $s$ hadron decays :: Leptonic tree-level decays :: $K\to \ell\nu$ :: $K^+\to \mu^+\nu_\mu$')
flavio.classes.Prediction(_obs_name, r_klnu_fct('K+'))

# for the pion decay, we only need the branching ratio of pi->enu, as
# pi->munu is 100%!
_obs_name = "BR(pi+->enu)"
_obs = flavio.classes.Observable(_obs_name)
_process_tex = r"\pi^+\to e^+\nu"
_obs.set_description(r"Branching ratio of $" + _process_tex + r"$")
_obs.tex = r"$\text{BR}(" + _process_tex + r")$"
_process_taxonomy = r'Process :: Unflavoured meson decays :: Leptonic tree-level decays :: $\pi\to \ell\nu$ :: $' + _process_tex + r'$'
_obs.add_taxonomy(_process_taxonomy)
flavio.classes.Prediction(_obs_name, r_klnu_fct('pi+'))
