import flavio
import numpy as np
from collections import OrderedDict

def np_prediction(obs_name, wc_obj, *args, **kwargs):
    obs = flavio.classes.Observable.get_instance(obs_name)
    return obs.prediction_central(flavio.default_parameters, wc_obj, *args, **kwargs)

def sm_prediction(obs_name, *args, **kwargs):
    obs = flavio.classes.Observable.get_instance(obs_name)
    wc_sm = flavio.WilsonCoefficients()
    return obs.prediction_central(flavio.default_parameters, wc_sm, *args, **kwargs)

def sm_uncertainty(obs_name, *args, N=100, **kwargs):
    obs = flavio.classes.Observable.get_instance(obs_name)
    wc_sm = flavio.WilsonCoefficients()
    par_random = [flavio.default_parameters.get_random_all() for i in range(N)]
    all_pred = np.array([
        obs.prediction_par(par, wc_sm, *args, **kwargs)
        for par in par_random
    ])
    return np.std(all_pred)

def sm_error_budget(obs_name, *args, N=50, **kwargs):
    obs = flavio.classes.Observable.get_instance(obs_name)
    wc_sm = flavio.WilsonCoefficients()
    par_central = flavio.default_parameters.get_central_all()
    par_random = [flavio.default_parameters.get_random_all() for i in range(N)]

    # Step 1: determine the parameters the observable depends on at all.
    # to this end, compute the observables once for each parameter with a
    # random value for this parameter but central values for all other
    # parameters. If the prediction is equal to the central prediction, the
    # observable does not depend on the parameter!
    pred_central = obs.prediction_par(par_central, wc_sm, *args, **kwargs)
    dependent_par = []
    for k in par_central.keys():
        par_tmp = par_central.copy()
        par_tmp[k] = par_random[0][k]
        pred_tmp = obs.prediction_par(par_tmp, wc_sm, *args, **kwargs)
        if pred_tmp != pred_central:
            dependent_par.append(k)

    # Step 2: for each of the dependent parameters, determine the error
    # analogous to the sm_uncertainty function. Normalize to the central
    # prediction (so relative errors are returned)
    individual_errors = {}
    def make_par_random(key, par_random):
        par_tmp = par_central.copy()
        par_tmp[key] = par_random[key]
        return par_tmp
    for p in dependent_par:
        par_random_p = [make_par_random(p, pr) for pr in par_random]
        all_pred = np.array([
            obs.prediction_par(par, wc_sm, *args, **kwargs)
            for par in par_random_p
        ])
        individual_errors[p] = np.std(all_pred)/pred_central
    return individual_errors


    # all_pred = np.array([
    #     obs.prediction_par(par, wc_sm)
    #     for par in par_random
    # ])
    # par_keys = par_random[0].keys()
    # par_random_arr = np.array([np.array([p[key] for key in par_keys]) for p in par_random])
    # corr_dict = { list(par_keys)[i]: np.corrcoef(par_random_arr[:,i],all_pred)[0,1] for i in range(len(par_keys)) }
    # return par_keys, par_random_arr, all_pred#OrderedDict(sorted(corr_dict.items(), key=lambda t: t[1]))
