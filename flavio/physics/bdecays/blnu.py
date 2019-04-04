r"""Functions for $B\to\ell\nu$ decays."""

import flavio
from math import pi
from flavio.physics.bdecays.common import meson_quark

def br_plnu_general(wc, par, Vij, P, qiqj, lep, nu, mq1, mq2, delta=0):
    r"""Branching ratio of general $P^+\to\ell^+\nu_\ell$ decay.

    `Vij` is the appropriate CKM matrix element.
    `mq1` and `mq2` are the masses of the two quarks forming the meson $P$.
    `delta` (detaults to 0) is a correction factor to account for different
    experimental treatment of electromagnetic effects, for instance.
    """
    ml = par['m_'+lep]
    mP = par['m_'+P]
    GF = par['GF']
    tau = par['tau_'+P]
    f = par['f_'+P]
    # Wilson coefficient dependence
    qqlnu = qiqj + lep + 'nu' + nu
    rWC = (wc['CVL_'+qqlnu] - wc['CVR_'+qqlnu]) + mP**2/ml/(mq1 + mq2) * (wc['CSR_'+qqlnu] - wc['CSL_'+qqlnu])
    N = tau * GF**2 * f**2 / (8*pi) * mP * ml**2  * (1 - ml**2/mP**2)**2
    return N * abs(Vij)**2 * abs(rWC)**2 * (1 + delta)


def br_blnu(wc_obj, par, B, lep):
    return sum([_br_blnu(wc_obj,par,B,lep,nu) for nu in ['e', 'mu', 'tau']])


def _br_blnu(wc_obj, par, B, lep, nu):
    r"""Branching ratio of $B_q\to\ell^+\nu_\ell$."""
    bq = meson_quark[B]
    # CKM element
    if bq == 'bc':
        Vxb = flavio.physics.ckm.get_ckm(par)[1,2]
    elif bq == 'bu':
        Vxb = flavio.physics.ckm.get_ckm(par)[0,2]
    # renormalization scale
    scale = flavio.config['renormalization scale']['bll']
    # Wilson coefficients
    wc = wc_obj.get_wc(bq + lep + 'nu' + nu, scale, par)
    # add SM contribution to Wilson coefficient
    if lep == nu:
        wc['CVL_'+bq+lep+'nu'+nu] += flavio.physics.bdecays.wilsoncoefficients.get_CVLSM(par, scale, nf=5)
    mb = flavio.physics.running.running.get_mb(par, scale)
    if B == 'B+':
        mq = 0  # neglecting up quark mass
    elif B == 'Bc':
        mq = flavio.physics.running.running.get_mc(par, scale)
    return br_plnu_general(wc, par, Vxb, B, bq, lep, nu, mb, mq, delta=0)

# function returning function needed for prediction instance
def br_blnu_fct(B, lep):
    def f(wc_obj, par):
        return br_blnu(wc_obj, par, B, lep)
    return f

# Observable and Prediction instances
_tex = {'e': 'e', 'mu': '\mu', 'tau': r'\tau'}
_tex_B = {'B+': r'B^+', 'Bc': r'B_c'}
for l in ['e', 'mu', 'tau']:
    for B in ['B+', 'Bc']:
        _process_tex = _tex_B[B] + r"\to "+_tex[l]+r"^+\nu"
        _process_taxonomy = r'Process :: $b$ hadron decays :: Leptonic tree-level decays :: $B\to \ell\nu$ :: $' + _process_tex + r"$"

        _obs_name = "BR("+B+"->"+l+"nu)"
        _obs = flavio.classes.Observable(_obs_name)
        _obs.set_description(r"Branching ratio of $" + _process_tex + r"$")
        _obs.tex = r"$\text{BR}(" + _process_tex + r")$"
        _obs.add_taxonomy(_process_taxonomy)
        flavio.classes.Prediction(_obs_name, br_blnu_fct(B, l))
