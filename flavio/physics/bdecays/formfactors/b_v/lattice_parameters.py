import pkgutil
import csv
import numpy as np
from flavio.classes import Parameter
from flavio.statistics.probability import MultivariateNormalDistribution

def csv_to_dict(filename):
    f = pkgutil.get_data('flavio.physics', filename)
    datareader = csv.reader(f.decode('utf-8').splitlines(), dialect='excel-tab')
    res = {}
    for line in datareader:
        if len(line) == 2: # for the central values
            # do not read the results for the c parameters - they are not needed.
            if line[0].split('_')[1][0]=='c':
                continue
            res[line[0]] = float(line[1])
        elif len(line) == 3: # for the covariance
            # do not read the results for the c parameters - they are not needed.
            if line[0].split('_')[1][0]=='c' or line[1].split('_')[1][0]=='c':
                continue
            res[(line[0],line[1])] = float(line[2])
    return res


def load_parameters(file_res, file_cov, process, constraints):
    implementation_name = process + ' SSE'
    res_dict = csv_to_dict(file_res)
    cov_dict = csv_to_dict(file_cov)
    keys_sorted = sorted(res_dict.keys())
    res = [res_dict[k] for k in keys_sorted]
    # M -> M + M^T - diag(M) since the dictionary contains only the entries above the diagonal
    cov = ( np.array([[ cov_dict.get((k,m),0) for m in keys_sorted] for k in keys_sorted])
          + np.array([[ cov_dict.get((m,k),0) for m in keys_sorted] for k in keys_sorted])
          - np.diag([ cov_dict[(k,k)] for k in keys_sorted]) )
    parameter_names = [implementation_name + ' ' + coeff_name for coeff_name in keys_sorted]
    for parameter_name in parameter_names:
        try: # check if parameter object already exists
            p = Parameter.get_instance(parameter_name)
        except: # otherwise, create a new one
            p = Parameter(parameter_name)
        else: # if parameter exists, remove existing constraints
            constraints.remove_constraints(parameter_name)
    constraints.add_constraint(parameter_names,
            MultivariateNormalDistribution(central_value=res, covariance=cov ))


def lattice_load(constraints):
    load_parameters('data/arXiv-1501-00367v2/av_sl_results.d',
                    'data/arXiv-1501-00367v2/av_sl_covariance.d',
                    'B->K*', constraints)
    load_parameters('data/arXiv-1501-00367v2/av_ls_results.d',
                    'data/arXiv-1501-00367v2/av_ls_covariance.d',
                    'Bs->K*', constraints)
    load_parameters('data/arXiv-1501-00367v2/av_ss_results.d',
                    'data/arXiv-1501-00367v2/av_ss_covariance.d',
                    'Bs->phi', constraints)
    load_parameters('data/arXiv-1501-00367v2/t_sl_results.d',
                    'data/arXiv-1501-00367v2/t_sl_covariance.d',
                    'B->K*', constraints)
    load_parameters('data/arXiv-1501-00367v2/t_ls_results.d',
                    'data/arXiv-1501-00367v2/t_ls_covariance.d',
                    'Bs->K*', constraints)
    load_parameters('data/arXiv-1501-00367v2/t_ss_results.d',
                    'data/arXiv-1501-00367v2/t_ss_covariance.d',
                    'Bs->phi', constraints)
