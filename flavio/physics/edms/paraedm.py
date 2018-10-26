"""Functions for EDMs of paramagnetic atoms and molecules."""

import flavio
from flavio.physics.edms.common import edm_f
from flavio.physics.edms.slcouplings import CS
from math import pi, sqrt


atoms = {
    'Tl': {
        'Z': 3, 'N': 4, 'tex': r'\text{Tl}', 'name': 'Thallium'
    },
}


molecules = {
    'YbF': {
        'Z': 1, 'N': 1, 'tex': r'\text{HfF}', 'name': 'Ytterbium fluoride'
    },
    'HfF': {
        'Z': 1, 'N': 1, 'tex': r'\text{HfF}', 'name': 'Hafnium fluoride'
    },
    'ThO': {
        'Z': 1, 'N': 1, 'tex': r'\text{HfF}', 'name': 'Thorium monoxide'
    },
}


def de(wc, par, scale):
    return edm_f(f='e', par=par, wc=wc, scale=scale, eft='WET-3')


def omega_para(wc_obj, par, molecule):
    wc = wc_obj.get_wc('dF=0', scale=2, par=par, eft='WET-3', basis='flavio')
    a_de = par['alpha_de({})'.format(molecule)]
    a_CS = par['alpha_CS({})'.format(molecule)]
    Z = molecules[molecule]['Z']
    N = molecules[molecule]['N']
    return a_de * de(wc, par, scale=2) +  a_CS * CS(wc, par, scale=2, Z=Z, N=N)


def d_para(wc_obj, par, atom):
    wc = wc_obj.get_wc('dF=0', scale=2, par=par, eft='WET-3', basis='flavio')
    a_de = par['alpha_de({})'.format(atom)]
    a_CS = par['alpha_CS({})'.format(atom)]
    Z = atoms[atom]['Z']
    N = atoms[atom]['N']
    return a_de * de(wc, par, scale=2) +  a_CS * CS(wc, par, scale=2, Z=Z, N=N)


# Observable and Prediction instances

def make_obs_d(symbol, texsymbol, name):
    _obs_name = "d_{}".format(symbol)
    _obs = flavio.classes.Observable(name=_obs_name)
    _obs.set_description(r"Electric dipole moment of {}".format(name))
    _obs.tex = r"$d_{}$".format(texsymbol)
    _obs.add_taxonomy(r'Process :: Dipole moments :: Atomic electric dipole moments :: $d_{}$'.format(texsymbol))
    flavio.classes.Prediction(_obs_name, lambda wc_obj, par: d_para(wc_obj, par, symbol))


def make_obs_omega(symbol, texsymbol, name):
    _obs_name = "omega_{}".format(symbol)
    _obs = flavio.classes.Observable(name=_obs_name)
    _obs.set_description(r"P- and T-violating energy shift in {}".format(name))
    _obs.tex = r"$\omega_{}$".format(texsymbol)
    _obs.add_taxonomy(r'Process :: Dipole moments :: Molecular energy shifts :: $d_{}$'.format(texsymbol))
    flavio.classes.Prediction(_obs_name, lambda wc_obj, par: omega_para(wc_obj, par, symbol))


for k, v in molecules.items():
    make_obs_omega(k, v['tex'], v['name'])


for k, v in atoms.items():
    make_obs_d(k, v['tex'], v['name'])
