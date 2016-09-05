r"""Functions for $B\to\ell\nu$ decays."""

import flavio
from math import pi

def br_plnu_general(wc, par, Vij, P, qiqj, lep, delta=0):
    r"""Branching ratio of general $P^+\to\ell^+\nu_\ell$ decay.

    `Vij` is the appropriate CKM matrix element.
    `delta` (detaults to 0) is a correction factor to account for different
    experimental treatment of electromagnetic effects, for instance.
    """
    ml = par['m_'+lep]
    mP = par['m_'+P]
    GF = par['GF']
    tau = par['tau_'+P]
    f = par['f_'+P]
    # Wilson coefficient dependence
    qqlnu = qiqj + lep + 'nu'
    rWC = (wc['CV_'+qqlnu] - wc['CVp_'+qqlnu]) + mP**2/ml * (wc['CS_'+qqlnu] - wc['CSp_'+qqlnu])
    N = tau * GF**2 * f**2 / (8*pi) * mP * ml**2  * (1 - ml**2/mP**2)**2
    return N * abs(Vij)**2 * abs(rWC)**2 * (1 + delta)

def br_blnu(wc_obj, par, lep):
    r"""Branching ratio of $B^+\to\ell^+\nu_\ell$."""
    # CKM element
    Vub = flavio.physics.ckm.get_ckm(par)[0,2]
    # renormalization scale
    scale = flavio.config['renormalization scale']['bll']
    # Wilson coefficients
    wc = wc_obj.get_wc('bu' + lep + 'nu', scale, par)
    # add SM contribution to Wilson coefficient
    wc['CV_bu'+lep+'nu'] += flavio.physics.bdecays.wilsoncoefficients.get_CVSM(par, scale, nf=5)
    return br_plnu_general(wc, par, Vub, 'B+', 'bu', lep, delta=0)

# function returning function needed for prediction instance
def br_blnu_fct(lep):
    def f(wc_obj, par):
        return br_blnu(wc_obj, par, lep)
    return f

# Observable and Prediction instances
_tex = {'e': 'e', 'mu': '\mu', 'tau': r'\tau'}
for l in ['e', 'mu', 'tau']:
    _obs_name = "BR(B+->"+l+"nu)"
    _obs = flavio.classes.Observable(_obs_name)
    _obs.set_description(r"Branching ratio of $B^+\to "+_tex[l]+r"^+\nu_"+_tex[l]+r"$")
    _obs.tex = r"$\text{BR}(B^+\to "+_tex[l]+r"^+\nu_"+_tex[l]+r")$"
    flavio.classes.Prediction(_obs_name, br_blnu_fct(l))
