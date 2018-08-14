r"""Functions for $D\to\ell\nu$."""

import flavio
from flavio.physics.bdecays.blnu import br_plnu_general
from math import pi, log

def br_Dlnu(wc_obj, par, P, lep):
    r"""Branching ratio of $D^+\to\ell^+\nu_\ell$."""
    return sum([ _br_Dlnu(wc_obj,par,P,lep,nu) for nu in ['e','mu','tau']])

def _br_Dlnu(wc_obj, par, P, lep, nu):
    # CKM element
    if P=='D+':
        Vij = flavio.physics.ckm.get_ckm(par)[1,0].conj() # Vcd*
        qiqj = 'dc'
    elif P=='Ds':
        Vij = flavio.physics.ckm.get_ckm(par)[1,1].conj() # Vcs*
        qiqj = 'sc'
    scale = flavio.config['renormalization scale']['dll']
    # Wilson coefficients
    wc = wc_obj.get_wc(qiqj + lep + 'nu' + nu, scale, par, nf_out=4)
    # add SM contribution to Wilson coefficient
    if lep == nu:
        wc['CVL_'+qiqj+lep+'nu' + nu] += flavio.physics.bdecays.wilsoncoefficients.get_CVLSM(par, scale, nf=4)
    mb = flavio.physics.running.running.get_mb(par, scale)
    mc = flavio.physics.running.running.get_mc(par, scale)
    return br_plnu_general(wc, par, Vij, P, qiqj, lep, nu, mb, mc, delta=0)

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
    _process_tex = _had[D]+r"\to "+_lep[l]+r"^+\nu_"+_lep[l]
    _process_taxonomy = r'Process :: $c$ hadron decays :: Leptonic tree-level decays :: $D\to \ell\nu$ :: $'

    _obs_name = "BR("+D+"->"+l+"nu)"
    _obs = flavio.classes.Observable(_obs_name)
    _obs.set_description(r"Branching ratio of $"+_process_tex+r"$")
    _obs.tex = r"$\text{BR}("+_process_tex+r")$"
    _obs.add_taxonomy(_process_taxonomy + _process_tex + r'$')
    flavio.classes.Prediction(_obs_name, br_Dlnu_fct(D, l))
