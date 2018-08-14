r"""Electric dipole moment of the neutron."""

import flavio
from flavio.config import config
from math import pi, sqrt


def nedm_wceff(wc_obj, par):
    scale = config['renormalization scale']['nEDM']
    v2 = 1 / (sqrt(2) * par['GF'])
    md = flavio.physics.running.running.get_md(par, scale, nf_out=3)
    mu = flavio.physics.running.running.get_mu(par, scale, nf_out=3)
    alpha = flavio.physics.running.running.get_alpha(par, scale, nf_out=3)
    e = sqrt(4 * pi * alpha['alpha_e'])
    gs = sqrt(4 * pi * alpha['alpha_s'])
    wc = wc_obj.get_wc('dF=0', scale=2, par=par, eft='WET-3', basis='flavio')
    wceff = {}
    pre = 4 * par['GF'] / sqrt(2) / (8 * pi**2)
    wceff['edm_d'] = pre * e * md * wc['C7_dd'].imag
    wceff['edm_u'] = pre * e * mu * wc['C7_uu'].imag
    wceff['cedm_d'] = pre * gs * md * wc['C8_dd'].imag
    wceff['cedm_u'] = pre * gs * mu * wc['C8_uu'].imag
    wceff['Gtilde'] = wc['CGtilde']
    return wceff


def nedm(wc_obj, par):
    wceff = nedm_wceff(wc_obj, par)
    d = 0
    # quark EDM & CEDM contributions
    d += par['nEDM gT_d'] * wceff['edm_d']
    d += par['nEDM gT_u'] * wceff['edm_u']
    d += par['nEDM ~rho_d'] * wceff['cedm_d']
    d += par['nEDM ~rho_u'] * wceff['cedm_u']
    d += par['nEDM beta_G'] * wceff['Gtilde']
    return abs(d)


# Observable and Prediction instances

_obs_name = "d_n"
_obs = flavio.classes.Observable(name=_obs_name)
_obs.set_description(r"Electric dipole moment of the neutron")
_obs.tex = r"$d_n$"
_obs.add_taxonomy(r'Process :: Electric dipole moments :: Nucleons :: $d_n$')
flavio.classes.Prediction(_obs_name, nedm)
