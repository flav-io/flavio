r"""Electric dipole moment of the neutron."""

import flavio
from .common import edm_f, cedm_f


def nedm_wceff(wc_obj, par):
    wc = wc_obj.get_wc('dF=0', scale=2, par=par, eft='WET-3', basis='flavio')
    wceff = {}
    opt = dict(par=par, wc=wc, scale=2, eft='WET-3')
    wceff['edm_d'] = edm_f(f='d', **opt)
    wceff['edm_u'] = edm_f(f='u', **opt)
    wceff['edm_s'] = edm_f(f='s', **opt)
    wceff['cedm_d'] = cedm_f(f='d', **opt)
    wceff['cedm_u'] = cedm_f(f='u', **opt)
    wceff['cedm_s'] = cedm_f(f='s', **opt)
    wceff['Gtilde'] = wc['CGtilde']
    return wceff


def nedm(wc_obj, par):
    wceff = nedm_wceff(wc_obj, par)
    d = 0
    # quark EDM & CEDM contributions
    d += par['gT_u'] * wceff['edm_d']  # u<->d due to proton<->neutron!
    d += par['gT_d'] * wceff['edm_u']  # u<->d due to proton<->neutron!
    d += par['gT_s'] * wceff['edm_s']
    d += par['nEDM ~rho_d'] * wceff['cedm_d']
    d += par['nEDM ~rho_u'] * wceff['cedm_u']
    d += par['nEDM ~rho_s'] * wceff['cedm_s']
    d += par['nEDM beta_G'] * wceff['Gtilde']
    return abs(d)


# Observable and Prediction instances

_obs_name = "d_n"
_obs = flavio.classes.Observable(name=_obs_name)
_obs.set_description(r"Electric dipole moment of the neutron")
_obs.tex = r"$d_n$"
_obs.add_taxonomy(r'Process :: Dipole moments :: Nucleon electric dipole moments :: $d_n$')
flavio.classes.Prediction(_obs_name, nedm)
