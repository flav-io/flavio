import pkgutil
import csv
import numpy as np
from flavio.classes import Parameter
from flavio.statistics.probability import MultivariateNormalDistribution

def csv_to_dict(filename):
    f = pkgutil.get_data('flavio.physics', filename)
    datareader = csv.reader(f.decode('utf-8').splitlines(), delimiter=' ', skipinitialspace=True)
    res = {}
    for line in datareader:
        if len(line) == 2: # for the central values
            res[line[0]] = float(line[1])
        elif len(line) == 3: # for the covariance
            res[(line[0],line[1])] = float(line[2])
    return res

ffname_dict = {
 'f0': 'fVt',
 'fperp': 'fVperp',
 'fplus': 'fV0',
 'g0': 'fAt',
 'gperp': 'fAperp',
 'gplus': 'fA0',
 'gpp': 'fA0', # gpp means g+ and gperp
 'hperp': 'fTperp',
 'hplus': 'fT0',
 'htildeperp': 'fT5perp',
 'htildeplus': 'fT50',
 'htildepp': 'fT50', # htildepp means htilde+ and htildeperp
 }

tex_a = {'a0': 'a_0', 'a1': 'a_1', 'a2': 'a_2', }
tex_ff = {
 'fA0': r'f^A_0', 'fAperp': r'f^A_\perp', 'fAt': r'f^A_t',
 'fT0': r'f^T_0', 'fT50': r'f^{T5}_0', 'fT5perp': r'f^{T5}_\perp',
 'fTperp': r'f^T_\perp', 'fV0': r'f^V_0', 'fVperp': r'f^V_\perp',
 'fVt': r'f^V_t'
 }

def translate_parameters(name):
    """Function to translate the parameter names from the ones used in the
    data files (e.g. 'a0_fplus') to the ones used in flavio (e.g. 'a0_fV0')."""
    part1 = name[0:3]
    part2 = name[3:]
    return part1 + ffname_dict[part2]

def load_parameters(file_res, file_cov, process, constraints):
    implementation_name = process + ' SSE'
    res_dict = csv_to_dict(file_res)
    cov_dict = csv_to_dict(file_cov)
    keys_sorted = sorted(res_dict.keys())
    res = [res_dict[k] for k in keys_sorted]
    cov = np.array([[ cov_dict.get((k,m),0) for m in keys_sorted] for k in keys_sorted])
    parameter_names = [implementation_name + ' ' + translate_parameters(coeff_name) for coeff_name in keys_sorted]
    for parameter_name in parameter_names:
        try: # check if parameter object already exists
            p = Parameter[parameter_name]
        except: # otherwise, create a new one
            p = Parameter(parameter_name)
            _tex_a = tex_a[parameter_name.split(' ')[-1].split('_')[0]]
            _tex_ff = tex_ff[parameter_name.split(' ')[-1].split('_')[-1]]
            p.tex = r'$' + _tex_a + r'^{' + _tex_ff + r'}$'
            p.description = r'SSE form factor parametrization coefficient $' + _tex_a + r'$ of $' + _tex_ff + r'$'
        else: # if parameter exists, remove existing constraints
            constraints.remove_constraint(parameter_name)
    constraints.add_constraint(parameter_names,
            MultivariateNormalDistribution(central_value=res, covariance=cov ))


def lattice_load_nominal(constraints):
    load_parameters('data/arXiv-1602-01399v1/LambdabLambda_results.dat',
                    'data/arXiv-1602-01399v1/LambdabLambda_covariance.dat',
                    'Lambdab->Lambda', constraints)

def lattice_load_ho(constraints):
    load_parameters('data/arXiv-1602-01399v1/LambdabLambda_HO_results.dat',
                    'data/arXiv-1602-01399v1/LambdabLambda_HO_covariance.dat',
                    'Lambdab->Lambda', constraints)
