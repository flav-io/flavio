import numpy as np
import json
import pkgutil
from flavio.classes import Parameter
from flavio.statistics.probability import MultivariateNormalDistribution


FFs = ["A0", "A1", "A12", "V", "T1", "T2", "T23"]
ai = ["a0", "a1", "a2"]
ff_a = [(ff, a) for ff in FFs for a in ai]
a_ff_string = [a + '_' + ff for ff in FFs for a in ai]


tex_a = {'a0': 'a_0', 'a1': 'a_1', 'a2': 'a_2', }
tex_ff = {'A0': 'A_0', 'A1': 'A_1', 'A12': r'A_{12}', 'V': 'V', 'T1': 'T_1', 'T2': 'T_2', 'T23': r'T_{23}', }


def get_ffpar(filename):
    f = pkgutil.get_data('flavio.physics', filename)
    data = json.loads(f.decode('utf-8'))
    central = np.array([data['central'][ff].get(a, np.nan) for ff, a in ff_a])
    unc = np.array([data['uncertainty'][ff].get(a, np.nan) for ff, a in ff_a])
    corr = np.array([[data['correlation'][ff1 + ff2].get(a1 + a2, np.nan) for ff1, a1 in ff_a] for ff2, a2 in ff_a])
    # delete the parameters a0_A12 and a0_T1, which are instead fixed
    # using the exact kinematical relations, cf. eq. (16) of arXiv:1503.05534
    pos_a0_A12 = ff_a.index(('A12', 'a0'))
    pos_a0_T2 = ff_a.index(('T2', 'a0'))
    central = np.delete(central, [pos_a0_A12, pos_a0_T2])
    unc = np.delete(unc, [pos_a0_A12, pos_a0_T2])
    corr = np.delete(corr, [pos_a0_A12, pos_a0_T2], axis=0)
    corr = np.delete(corr, [pos_a0_A12, pos_a0_T2], axis=1)
    return [central, unc, corr]


def load_parameters(filename, process, constraints):
    implementation_name = process + ' BSZ'
    parameter_names = [implementation_name + ' ' + coeff_name for coeff_name in a_ff_string]
    # a0_A0 and a0_T2 are not treated as independent parameters!
    parameter_names.remove(implementation_name + ' a0_A12')
    parameter_names.remove(implementation_name + ' a0_T2')
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


# Resonance masses used in arXiv:1503.05534
resonance_masses_bsz = {
    'B->K*': {
        'm0': 5.367,
        'm1-': 5.415,
        'm1+': 5.830,
    },
    'B->rho': {
        'm0': 5.279,
        'm1-': 5.324,
        'm1+': 5.716,
    },
    'B->omega': {
        'm0': 5.279,
        'm1-': 5.324,
        'm1+': 5.716,
    },
    'Bs->phi': {
        'm0': 5.367,
        'm1-': 5.415,
        'm1+': 5.830,
    },
    'Bs->K*': {
        'm0': 5.279,
        'm1-': 5.324,
        'm1+': 5.716,
    },
}

# Resonance masses used in arXiv:1811.00983
resonance_masses_gkvd = {
    'B->K*': {
        'm0': 5.336,
        'm1-': 5.412,
        'm1+': 5.829,
    },
    'B->rho': {
        'm0': 5.279,
        'm1-': 5.325,
        'm1+': 5.724,
    },
    'B->D*': {
        'm0': 6.275,
        'm1-': 6.330,
        'm1+': 6.767,
    },
}


def transition_filename(tr):
    """Get the part of the filename specifying the transition (e.g. BKstar)
    from a transition string (e.g. B->K*)."""
    return tr.replace('->', '').replace('*', 'star')


def bsz_load(version, fit, transitions, constraints):
    """Load the form factor parameters given in arXiv:1503.05534"""
    for tr in transitions:
        for m, v in resonance_masses_bsz[tr].items():
            constraints.set_constraint('{} BCL {}'.format(tr, m), v)
        filename = 'data/arXiv-1503-05534{}/{}_{}.json'.format(version, transition_filename(tr), fit)
        load_parameters(filename, tr, constraints)


def bsz_load_v1_lcsr(constraints):
    bsz_load('v1', 'LCSR', ('B->K*', 'B->omega', 'B->rho', 'Bs->phi', 'Bs->K*'), constraints)

def bsz_load_v1_combined(constraints):
    bsz_load('v1', 'LCSR-Lattice', ('B->K*', 'Bs->phi', 'Bs->K*'), constraints)

def bsz_load_v2_lcsr(constraints):
    bsz_load('v2', 'LCSR', ('B->K*', 'B->omega', 'B->rho', 'Bs->phi', 'Bs->K*'), constraints)

def bsz_load_v2_combined(constraints):
    bsz_load('v2', 'LCSR-Lattice', ('B->K*', 'Bs->phi', 'Bs->K*'), constraints)


def gkvd_load(version, fit, transitions, constraints):
    """Load the form factor parameters given in arXiv:1811.00983"""
    for tr in transitions:
        for m, v in resonance_masses_gkvd[tr].items():
            constraints.set_constraint('{} BCL {}'.format(tr, m), v)
        filename = 'data/arXiv-1811-00983{}/{}_{}.json'.format(version, transition_filename(tr), fit)
        load_parameters(filename, tr, constraints)
