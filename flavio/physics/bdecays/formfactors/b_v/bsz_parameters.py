import numpy as np
import json
import pkgutil
from flavio.classes import Parameter, MultivariateNormalDistribution

FFs = ["A0","A1","A12","V","T1","T2","T23"]
ai = ["a0","a1","a2"]
ff_a  = [(ff,a) for ff in FFs for a in ai]
a_ff_string  = [a + '_' + ff for ff in FFs for a in ai]

def get_ffpar(filename):
    f = pkgutil.get_data('flavio.physics', filename)
    data = json.loads(f.decode('utf-8'))
    central = np.array([data['central'][ff][a] for ff, a in ff_a])
    unc = np.array([data['uncertainty'][ff][a] for ff, a in ff_a])
    corr = np.array([[data['correlation'][ff1 + ff2][a1 + a2] for ff1, a1 in ff_a] for ff2, a2 in ff_a])
    # delete the parameters a0_A0 and a0_T2, which are instead fixed
    # using the exact kinematical relations, cf. eq. (16) of arXiv:1503.05534
    pos_a0_A0 = ff_a.index(('A0', 'a0'))
    pos_a0_T2 = ff_a.index(('T2', 'a0'))
    central = np.delete(central, [pos_a0_A0, pos_a0_T2])
    unc = np.delete(unc, [pos_a0_A0, pos_a0_T2])
    corr = np.delete(corr, [pos_a0_A0, pos_a0_T2], axis=0)
    corr = np.delete(corr, [pos_a0_A0, pos_a0_T2], axis=1)
    return [central, unc, corr]

def load_parameters(filename, process, constraints):
    implementation_name = process + ' BSZ3'
    parameter_names = [implementation_name + ' ' + coeff_name for coeff_name in a_ff_string]
    # a0_A0 and a0_T2 are not treated as independent parameters!
    parameter_names.remove(implementation_name + ' a0_A0')
    parameter_names.remove(implementation_name + ' a0_T2')
    for parameter_name in parameter_names:
        try: # check if parameter object already exists
            p = Parameter.get_instance(parameter_name)
        except: # otherwise, create a new one
            p = Parameter(parameter_name)
        else: # if parameter exists, remove existing constraints
            constraints.remove_constraints(parameter_name)
    [central, unc, corr] = get_ffpar(filename)
    constraints.add_constraint(parameter_names,
            MultivariateNormalDistribution(central_value=central, covariance=np.outer(unc, unc)*corr) )


def bsz_load_v1_lcsr(constraints):
    load_parameters('data/arXiv-1503-05534v1/BKstar_LCSR.json', 'B->K*', constraints)
    load_parameters('data/arXiv-1503-05534v1/Bomega_LCSR.json', 'B->omega', constraints)
    load_parameters('data/arXiv-1503-05534v1/Brho_LCSR.json', 'B->rho', constraints)
    load_parameters('data/arXiv-1503-05534v1/Bsphi_LCSR.json', 'Bs->phi', constraints)
    load_parameters('data/arXiv-1503-05534v1/BsKstar_LCSR.json', 'Bs->K*', constraints)

def bsz_load_v1_combined(constraints):
    load_parameters('data/arXiv-1503-05534v1/BKstar_LCSR-Lattice.json', 'B->K*', constraints)
    load_parameters('data/arXiv-1503-05534v1/Bsphi_LCSR-Lattice.json', 'Bs->phi', constraints)
    load_parameters('data/arXiv-1503-05534v1/BsKstar_LCSR-Lattice.json', 'Bs->K*', constraints)
