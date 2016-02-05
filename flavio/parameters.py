import yaml
import pkgutil
from flavio.classes import *
import csv

default_parameters = Constraints()

f = pkgutil.get_data('flavio', 'data/parameters.yml')
parameters = yaml.load(f)

def _read_yaml_object(obj):
    parameters = yaml.load(obj)
    for parameter_name, value in parameters.items():
        p = Parameter(parameter_name)
    default_parameters.set_constraint(parameter_name, value)

def read_file(filename):
    """Read parameter values from a YAML file."""
    with open(filename, 'r') as f:
        _read_yaml_object(f)

# particles from the PDG data file whose mass we're interested in)
pdg_include = ['B(s)', 'B(s)*', 'B*+', 'B*0', 'B+', 'B0', 'D(s)', 'D(s)*', 'D+', 'D0',
                'H', 'J/psi(1S)', 'K(L)', 'K(S)', 'K*(892)+', 'K*(892)0', 'K+', 'K0',
                'Lambda', 'Lambda(b)', 'Omega',
                 'W', 'Z',  'b',  'c', 'd', 'e', 'eta', 'f(0)(980)',
                 'mu',  'phi(1020)', 'pi+', 'pi0', 'psi(2S)', 'rho(770)+', 'rho(770)0',
                 's', 't', 'tau', 'u']
# dictionary translating PDG particle names to the ones in the code.
pdg_translate = {
'B(s)': 'Bs',
'D(s)': 'Ds',
'B(s)*': 'Bs*',
'D(s)*': 'Ds*',
'J/psi(1S)': 'J/psi',
'K(L)': 'KL',
'K(S)': 'KS',
'K*(892)+': 'K*+',
'K*(892)0': 'K*0',
'phi(1020)': 'phi',
'rho(770)0': 'rho0',
'rho(770)+': 'rho+',
'f(0)(980)': 'f0',
"eta'(958)": "eta'",
'Higgs' : 'h', # this is necessary for the 2013 data file
'H' : 'h',
}

def _read_pdg_masswidth(filename):
    """Read the PDG mass and width table and return a dictionary.

    Parameters
    ----------
    filname : string
        Path to the PDG data file, e.g. 'data/pdg/mass_width_2015.mcd'

    Returns
    -------
    particles : dict
        A dictionary where the keys are the particle names with the charge
        appended in case of a multiplet with different masses, e.g. 't'
        for the top quark, 'K+' and 'K0' for kaons.
        The value of the dictionary is again a dictionary with the following
        keys:
        - 'id': PDG particle ID
        - 'mass': list with the mass, postitive and negative error in GeV
        - 'width': list with the width, postitive and negative error in GeV
        - 'name': same as the key
    """
    data = pkgutil.get_data('flavio.physics', filename)
    lines = data.decode('utf-8').splitlines()
    particles_by_name = {}
    for line in lines:
        if  line.strip()[0] == '*':
            continue
        mass = ((line[33:51]),(line[52:60]),(line[61:69]))
        mass = [float(m) for m in mass]
        width = ((line[70:88]),(line[89:97]),(line[98:106]))
        if  width[0].strip() == '':
            width = (0,0,0)
        else:
            width = [float(w) for w in width]
        ids = line[0:32].split()
        charges = line[107:128].split()[1].split(',')
        if len(ids) != len(charges):
            raise ValueError()
        for i in range(len(ids)):
            particle = {}
            particle_charge = charges[i].strip()
            particle[particle_charge] = {}
            particle[particle_charge]['id'] = ids[i].strip()
            particle[particle_charge]['mass']  = mass
            particle[particle_charge]['charge']  = particle_charge
            particle[particle_charge]['width'] = width
            particle_name = line[107:128].split()[0]
            particle[particle_charge]['name'] = particle_name
            if particle_name in particles_by_name.keys():
                particles_by_name[particle_name].update(particle)
            else:
                particles_by_name[particle_name] = particle
    result = { k + kk: vv for k, v in particles_by_name.items() for kk, vv in v.items() if len(v) > 1}
    result.update( { k: list(v.values())[0] for k, v in particles_by_name.items() if len(v) == 1} )
    return result


def read_pdg(year):
    """Read particle masses and widths from the PDG data file of a given year."""
    particles = _read_pdg_masswidth('data/pdg/mass_width_' + str(year) + '.mcd')
    for particle in pdg_include:
        parameter_name = 'm_' + pdg_translate.get(particle, particle) # translate if necessary
        try:
            # if parameter already exists, remove existing constraints on it
            m = Parameter.get_instance(parameter_name)
            m.remove_constraints()
        except KeyError:
            # otherwise, create it
            m = Parameter(parameter_name)
        m_central, m_right, m_left = particles[particle]['mass']
        m_left = abs(m_left) # make left error positive
        if m_right == m_left:
            m.add_constraint(NormalDistribution(m_central, m_right))
        else:
            m.add_constraint(AsymmetricNormalDistribution(m_central, right_deviation=m_right, left_deviation=m_left))
        if particles[particle]['width'][0] == 0: # 0 is for particles where the width is unknown (e.g. B*)
            continue
        G_central, G_right, G_left = particles[particle]['width']
        G_left = abs(G_left) # make left error positive
        parameter_name = 'tau_' + pdg_translate.get(particle, particle) # translate if necessary
        try:
            # if parameter already exists, remove existing constraints on it
            tau = Parameter.get_instance(parameter_name)
            tau.remove_constraints()
        except KeyError:
            # otherwise, create it
            tau = Parameter(parameter_name)
        tau_central = 1/G_central # life time = 1/width
        tau_left = G_left/G_central**2
        tau_right = G_right/G_central**2
        if tau_left == tau_right:
            tau.add_constraint(NormalDistribution(tau_central, tau_right))
        else:
            tau.add_constraint(AsymmetricNormalDistribution(tau_central, right_deviation=tau_right, left_deviation=tau_left))

# Read default parameters

## Read the parameters from the default YAML data file
_read_yaml_object(pkgutil.get_data('flavio', 'data/parameters.yml'))

## Read the parameters from the default PDG data file
read_pdg(2015)
