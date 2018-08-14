"""Main functions for user interaction. All of these are imported into the
top-level namespace."""

import flavio
import numpy as np
from collections import defaultdict
from multiprocessing import Pool

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
    par_random = [flavio.default_parameters.get_random_all() for i in range(N)]
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

def sm_covariance(obs_list, N=100, par_vary='all', **kwargs):
    """Get the covariance matrix of the Standard Model predictions for a
    list of observables.

    Parameters
    ----------

    - `obs_list`: a list of observables that should be given either as a string
    name (for observables that do not depend on any arguments) or as a tuple
    of a string and values for the arguements the observable depends on (e.g.
    the values of `q2min` and `q2max` for a binned observable)
    - `N` (optional): number of random evaluations of the observables.
    The relative accuracy of the uncertainties returned is given by $1/\sqrt{2N}$.
    - `par_vary`: a list of parameters to vary. Defaults to 'all', i.e. all
    parameters are varied according to their probability distributions.
    """
    wc_sm = flavio.physics.eft._wc_sm
    par_central_all = flavio.default_parameters.get_central_all()
    par_random_all = [flavio.default_parameters.get_random_all() for i in range(N)]
    def par_random_some(par_random, par_central):
        # take the central values for the parameters not to be varied
        par1 = {k: v for k, v in par_central.items() if k not in par_vary}
        # take the random values for the parameters to be varied
        par2 = {k: v for k, v in par_random.items() if k in par_vary}
        par1.update(par2) # merge them
        return par1
    if par_vary == 'all':
        par_random = par_random_all
    else:
        par_random = [par_random_some(par_random_all[i], par_central_all) for i in range(N)]
    def get_prediction(obs, par):
        if isinstance(obs, str):
             obs_obj = flavio.classes.Observable[obs]
             return obs_obj.prediction_par(par, wc_sm, **kwargs)
        elif isinstance(obs, tuple):
             obs_obj = flavio.classes.Observable[obs[0]]
             return obs_obj.prediction_par(par, wc_sm, *obs[1:], **kwargs)
    all_pred = np.array([
        [get_prediction(obs, par)
            for par in par_random
        ]
        for obs in obs_list
    ])
    return np.cov(all_pred)
