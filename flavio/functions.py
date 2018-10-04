"""Main functions for user interaction. All of these are imported into the
top-level namespace."""

import flavio
import numpy as np
from collections import defaultdict
from multiprocessing import Pool
from functools import partial
import warnings


def np_prediction(obs_name, wc_obj, *args, **kwargs):
    """Get the central value of the new physics prediction of an observable.

    Parameters
    ----------

    - `obs_name`: name of the observable as a string
    - `wc_obj`: an instance of `flavio.WilsonCoefficients`

    Additional arguments are passed to the observable and are necessary,
    depending on the observable (e.g. $q^2$-dependent observables).
    """
    obs = flavio.classes.Observable[obs_name]
    return obs.prediction_central(flavio.default_parameters, wc_obj, *args, **kwargs)

def sm_prediction(obs_name, *args, **kwargs):
    """Get the central value of the Standard Model prediction of an observable.

    Parameters
    ----------

    - `obs_name`: name of the observable as a string

    Additional arguments are passed to the observable and are necessary,
    depending on the observable (e.g. $q^2$-dependent observables).
    """
    obs = flavio.classes.Observable[obs_name]
    wc_sm = flavio.physics.eft._wc_sm
    return obs.prediction_central(flavio.default_parameters, wc_sm, *args, **kwargs)

def _obs_prediction_par(par, obs_name, wc_obj, *args, **kwargs):
    obs = flavio.classes.Observable.get_instance(obs_name)
    return obs.prediction_par(par, wc_obj, *args, **kwargs)

from functools import partial

def np_uncertainty(obs_name, wc_obj, *args, N=100, threads=1, **kwargs):
    """Get the uncertainty of the prediction of an observable in the presence
    of new physics.

    Parameters
    ----------

    - `obs_name`: name of the observable as a string
    - `wc_obj`: an instance of `flavio.WilsonCoefficients`
    - `N` (optional): number of random evaluations of the observable.
    The relative accuracy of the uncertainty returned is given by $1/\sqrt{2N}$.
    - `threads` (optional): if bigger than one, number of threads for parallel
    computation of the uncertainty.

    Additional arguments are passed to the observable and are necessary,
    depending on the observable (e.g. $q^2$-dependent observables).
    """
    par_random = flavio.default_parameters.get_random_all(size=N)
    par_random = [{k: v[i] for k, v in par_random.items()} for i in range(N)]
    if threads == 1:
        # not parallel
        all_pred = np.array([_obs_prediction_par(par, obs_name, wc_obj, *args, **kwargs) for par in par_random])
    else:
        # parallel
        pool = Pool(threads)
        # convert args to kwargs
        _kwargs = kwargs.copy()
        obs_args = flavio.Observable[obs_name].arguments
        for i, a in enumerate(args):
            _kwargs[obs_args[i]] = a
        all_pred = np.array(
                    pool.map(
                        partial(_obs_prediction_par,
                        obs_name=obs_name, wc_obj=wc_obj, **_kwargs),
                        par_random))
        pool.close()
        pool.join()
    return np.std(all_pred)

def sm_uncertainty(obs_name, *args, N=100, threads=1, **kwargs):
    """Get the uncertainty of the Standard Model prediction of an observable.

    Parameters
    ----------

    - `obs_name`: name of the observable as a string
    - `N` (optional): number of random evaluations of the observable.
    The relative accuracy of the uncertainty returned is given by $1/\sqrt{2N}$.
    - `threads` (optional): if bigger than one, number of threads for parallel
    computation of the uncertainty.

    Additional arguments are passed to the observable and are necessary,
    depending on the observable (e.g. $q^2$-dependent observables).
    """
    wc_sm = flavio.physics.eft._wc_sm
    return np_uncertainty(obs_name, wc_sm, *args, N=N, threads=threads, **kwargs)

class AwareDict(dict):
    """Generalization of dictionary that adds the key to the previously defined
    set `pcalled` upon getting an item."""

    def __init__(self, d):
        """Initialize the instance."""
        super().__init__(d)
        self.akeys = set()
        self.d = d

    def __getitem__(self, key):
        """Get an item, adding the key to the `pcalled` set."""
        self.akeys.add(key)
        return dict.__getitem__(self, key)

    def __copy__(self):
        cp = type(self)(self.d)
        cp.akeys = self.akeys
        return cp

    def copy(self):
        return self.__copy__()

def get_dependent_parameters_sm(obs_name, *args, **kwargs):
    """Get the set of parameters the SM prediction of the observable depends on."""
    obs = flavio.classes.Observable[obs_name]
    wc_sm = flavio.physics.eft._wc_sm
    par_central = flavio.default_parameters.get_central_all()
    apar_central = AwareDict(par_central)
    obs.prediction_par(apar_central, wc_sm, *args, **kwargs)
    # return all observed keys except the ones that don't actually correspond
    # to existing parameter names (this might happen by user functions modifying
    # the dictionaries)
    return {p for p in apar_central.akeys
            if p in flavio.Parameter.instances.keys()}

def sm_error_budget(obs_name, *args, N=50, **kwargs):
    """Get the *relative* uncertainty of the Standard Model prediction due to
    variation of individual observables.

    Parameters
    ----------

    - `obs_name`: name of the observable as a string
    - `N` (optional): number of random evaluations of the observable.
    The relative accuracy of the uncertainties returned is given by $1/\sqrt{2N}$.

    Additional arguments are passed to the observable and are necessary,
    depending on the observable (e.g. $q^2$-dependent observables).
    """
    obs = flavio.classes.Observable[obs_name]
    wc_sm = flavio.physics.eft._wc_sm
    par_central = flavio.default_parameters.get_central_all()
    par_random = [flavio.default_parameters.get_random_all() for i in range(N)]
    pred_central = obs.prediction_par(par_central, wc_sm, *args, **kwargs)

    # Step 1: determine the parameters the observable depends on at all.
    dependent_par = get_dependent_parameters_sm(obs_name, *args, **kwargs)

    # Step 2: group parameters if correlated
    par_constraint = {p: id(flavio.default_parameters._parameters[p][1]) for p in dependent_par}
    v = defaultdict(list)
    for key, value in par_constraint.items():
        v[value].append(key)
    dependent_par_lists = list(v.values())

    # Step 3: for each of the (groups of) dependent parameters, determine the error
    # analogous to the sm_uncertainty function. Normalize to the central
    # prediction (so relative errors are returned)
    individual_errors = {}
    def make_par_random(keys, par_random):
        par_tmp = par_central.copy()
        for key in keys:
            par_tmp[key] = par_random[key]
        return par_tmp
    for p in dependent_par_lists:
        par_random_p = [make_par_random(p, pr) for pr in par_random]
        all_pred = np.array([
            obs.prediction_par(par, wc_sm, *args, **kwargs)
            for par in par_random_p
        ])
        # for the dictionary key, use the list element if there is only 1,
        # otherwise use a tuple (which is hashable)
        if len(p) == 1:
            key = p[0]
        else:
            key = tuple(p)
        individual_errors[key] = np.std(all_pred)/abs(pred_central)
    return individual_errors


def _get_prediction_array_sm(par, obs_list):
    wc_sm = flavio.physics.eft._wc_sm
    def get_prediction_sm(obs, par):
        obs_dict = flavio.classes.Observable.argument_format(obs, 'dict')
        obs_obj = flavio.classes.Observable[obs_dict.pop('name')]
        return obs_obj.prediction_par(par, wc_sm, **obs_dict)
    return np.array([get_prediction_sm(obs, par) for obs in obs_list])


def sm_covariance(obs_list, N=100, par_vary='all', par_obj=None, threads=1,
                  **kwargs):
    r"""Get the covariance matrix of the Standard Model predictions for a
    list of observables.

    Parameters
    ----------

    - `obs_list`: a list of observables that should be given either as a string
    name (for observables that do not depend on any arguments) or as a tuple
    of a string and values for the arguements the observable depends on (e.g.
    the values of `q2min` and `q2max` for a binned observable)
    - `N` (optional): number of random evaluations of the observables.
    The relative accuracy of the uncertainties returned is given
    by $1/\sqrt{2N}$.
    - `par_vary` (optional): a list of parameters to vary. Defaults to 'all', i.e. all
    parameters are varied according to their probability distributions.
    - `par_obj` (optional): an instance of ParameterConstraints, defaults to
    flavio.default_parameters.
    - `threads` (optional): number of CPU threads to use for the computation.
    Defaults to 1, i.e. serial computation.
    """
    par_obj = par_obj or flavio.default_parameters
    par_central_all = par_obj.get_central_all()
    par_random_all = par_obj.get_random_all(size=N)

    def par_random_some(par_random, par_central):
        # take the central values for the parameters not to be varied (N times)
        par1 = {k: np.full(N, v) for k, v in par_central.items() if k not in par_vary}
        # take the random values for the parameters to be varied
        par2 = {k: v for k, v in par_random.items() if k in par_vary}
        par1.update(par2)  # merge them
        return par1

    if par_vary == 'all':
        par_random = par_random_all
        par_random = [{k: v[i] for k, v in par_random.items()} for i in range(N)]
    else:
        par_random = par_random_some(par_random_all, par_central_all)
        par_random = [{k: v[i] for k, v in par_random.items()} for i in range(N)]

    func_map = partial(_get_prediction_array_sm, obs_list=obs_list)
    if threads == 1:
        pred_map = map(func_map, par_random)
    else:
        pool = Pool(threads)
        pred_map = pool.map(func_map, par_random)
        pool.close()
        pool.join()
    all_pred = np.array(list(pred_map))
    return np.cov(all_pred.T)


def combine_measurements(observable, include_measurements=None,
                         **kwargs):
    """Combine all existing measurements of a particular observable.

    Returns a one-dimensional instance of `ProbabilityDistribution`.
    Correlations with other obersables are ignored.

    Parameters:

    - `observable`: observable name
    - `include_measurements`: iterable of measurement names to be included
      (default: all)

    Observable arguments have to be specified as keyword arguments, e.g.
    `combine_measurements('<dBR/dq2>(B+->Kmumu)', q2min=1, q2max=6)`.

    Note that this function returns inconsistent results (and a corresponding
    warning is issued) if an observable is constrained by more than one
    multivariate measurement.
    """
    if not kwargs:
        obs = observable
    else:
        args = flavio.Observable[observable].arguments
        obs = (observable, ) + tuple(kwargs[a] for a in args)
    constraints = []
    _n_multivariate = 0  # number of multivariate constraints
    for name, m in flavio.Measurement.instances.items():
        if include_measurements is not None and name not in include_measurements:
            continue
        if obs not in m.all_parameters:
            continue
        num, constraint = m._parameters[obs]
        if not np.isscalar(constraint.central_value):
            _n_multivariate += 1
            # for multivariate PDFs, reduce to 1D PDF
            exclude = tuple([i for i, _ in enumerate(constraint.central_value)
                             if i != num])  # exclude all i but num
            constraint1d = constraint.reduce_dimension(exclude=exclude)
            constraints.append(constraint1d)
        else:
            constraints.append(constraint)
    if _n_multivariate > 1:
        warnings.warn(("{} of the measurements of '{}' are multivariate. "
                       "This can lead to inconsistent results as the other "
                       "observables are profiled over. "
                       "To be consistent, you should perform a multivariate "
                       "combination that is not yet supported by `combine_measurements`."
                       ).format(_n_multivariate, obs))
    if not constraints:
        raise ValueError("No experimental measurements found for this observable.")
    elif len(constraints) == 1:
        return constraints[0]
    else:
        return flavio.statistics.probability.combine_distributions(constraints)
