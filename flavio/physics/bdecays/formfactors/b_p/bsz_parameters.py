import numpy as np
import json
import pkgutil
from flavio.classes import Parameter
from flavio.statistics.probability import MultivariateNormalDistribution


FFs = ["f+", "fT", "f0"]
ai = ["a0", "a1", "a2"]
ff_a = [(ff, a) for ff in FFs for a in ai]
a_ff_string = [a + '_' + ff for ff in FFs for a in ai]


tex_a = {'a0': 'a_0', 'a1': 'a_1', 'a2': 'a_2', }
tex_ff = {'f+': 'f_+', 'fT': 'f_T', 'f0': 'f_0', }


def get_ffpar(filename):
    f = pkgutil.get_data('flavio.physics', filename)
    data = json.loads(f.decode('utf-8'))
    central = np.array([data['central'][ff].get(a, np.nan) for ff, a in ff_a])
    unc = np.array([data['uncertainty'][ff].get(a, np.nan) for ff, a in ff_a])
    corr = np.array([[data['correlation'][ff1 + ff2].get(a1 + a2, np.nan) for ff1, a1 in ff_a] for ff2, a2 in ff_a])
    # delete the parameter a0_f0, which is instead fixed
    # using the exact kinematical relation f0(0) = f+(0)
    pos_a0_f0 = ff_a.index(('f0', 'a0'))
    central = np.delete(central, [pos_a0_f0])
    unc = np.delete(unc, [pos_a0_f0])
    corr = np.delete(corr, [pos_a0_f0], axis=0)
    corr = np.delete(corr, [pos_a0_f0], axis=1)
    return [central, unc, corr]


def load_parameters(filename, process, constraints):
    implementation_name = process + ' BSZ'
    parameter_names = [implementation_name + ' ' + coeff_name for coeff_name in a_ff_string]
    # a0_f0 is not treated as independent parameter!
    parameter_names.remove(implementation_name + ' a0_f0')
    for parameter_name in parameter_names:
        try:  # check if parameter object already exists
            p = Parameter[parameter_name]
        except KeyError:  # if not, create a new one
            p = Parameter(parameter_name)
            # get LaTeX representation of coefficient and form factor names
            _tex_a = tex_a[parameter_name.split(' ')[-1].split('_')[0]]
            _tex_ff = tex_ff[parameter_name.split(' ')[-1].split('_')[-1]]
            p.tex = r'$' + _tex_a + r'^{' + _tex_ff + r'}$'
            p.description = r'BSZ form factor parametrization coefficient $' + _tex_a + r'$ of $' + _tex_ff + r'$'
        else:  # if parameter exists, remove existing constraints
            constraints.remove_constraint(parameter_name)
    [central, unc, corr] = get_ffpar(filename)
    constraints.add_constraint(parameter_names,
        MultivariateNormalDistribution(central_value=central, covariance=np.outer(unc, unc)*corr))



# Resonance masses used in arXiv:1811.00983
resonance_masses_gkvd = {
    'B->K': {
        'm0': 5.630,
        'm+': 5.412,
    },
    'B->pi': {
        'm0': 5.540,
        'm+': 5.325,
    },
    'B->D': {
        'm0': 6.420,
        'm+': 6.330,
    },
}


def gkvd_load(version, fit, transitions, constraints):
    """Load the form factor parameters given in arXiv:1811.00983"""
    for tr in transitions:
        for m, v in resonance_masses_gkvd[tr].items():
            constraints.set_constraint('{} BCL {}'.format(tr, m), v)
        filename = 'data/arXiv-1811-00983{}/{}_{}.json'.format(version, tr.replace('->', ''), fit)
        load_parameters(filename, tr, constraints)
