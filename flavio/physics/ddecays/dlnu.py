r"""Functions for $D\to\ell\nu$."""

import flavio
from flavio.physics.bdecays.blnu import br_plnu_general
from math import pi, log

def br_Dlnu(wc_obj, par, P, lep):
    r"""Branching ratio of $D^+\to\ell^+\nu_\ell$."""
    # CKM element
    if P=='D+':
        Vij = flavio.physics.ckm.get_ckm(par)[1,0].conj() # Vcd*
        qiqj = 'dc'
    elif P=='Ds':
        Vij = flavio.physics.ckm.get_ckm(par)[1,1].conj() # Vcs*
        qiqj = 'sc'
    scale = flavio.config['renormalization scale']['dll']
    # Wilson coefficients
    wc = wc_obj.get_wc(qiqj + lep + 'nu', scale, par, nf_out=4)
    # add SM contribution to Wilson coefficient
    wc['CV_'+qiqj+lep+'nu'] += flavio.physics.bdecays.wilsoncoefficients.get_CVSM(par, scale, nf=4)
    return br_plnu_general(wc, par, Vij, P, qiqj, lep, delta=0)

# function returning function needed for prediction instance
def br_Dlnu_fct(P, lep):
    def f(wc_obj, par):
        return br_Dlnu(wc_obj, par, P, lep)
    return f

# Observable and Prediction instances

_lep = {'e': 'e', 'mu': r'\mu', 'tau': r'\tau'}
_had = {'D+': r'D^+', 'Ds': r'D_s',}

for D in _had:
  for l in _lep:
    _obs_name = "BR("+D+"->"+l+"nu)"
    _obs = flavio.classes.Observable(_obs_name)
    _obs.set_description(r"Branching ratio of $"+_had[D]+r"\to "+_lep[l]+r"^+\nu_"+_lep[l]+r"$")
    _obs.tex = r"$\text{BR}("+_had[D]+r"\to "+_lep[l]+r"^+\nu_"+_lep[l]+r")$"
    flavio.classes.Prediction(_obs_name, br_Dlnu_fct(D, l))
