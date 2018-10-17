import flavio
from collections import Counter
import warnings
import numpy as np
import voluptuous as vol
import pickle
from flavio.classes import NamedInstanceClass
from flavio.statistics.probability import NormalDistribution, MultivariateNormalDistribution
from flavio.io import instanceio as iio


class MeasurementLikelihood(iio.YAMLLoadable):
    """A `MeasurementLikelihood` provides a likelihood function from
    experimental measurements.

    Methods:

    - `get_predictions_par`: Return a dictionary of SM predictions for the
    observables of interest
    - `log_likelihood_pred`: The likelihood as a function of the predictions
    - `log_likelihood_par`: The likelihood as a function of parameters and
    Wilson coefficients

    Instances can be imported and exported from/to YAML using the `load`
    and `dump` methods.
    """

    _input_schema_dict = {
        'observables':  vol.All([iio.coerce_observable_tuple], iio.list_deduplicate),
        'exclude_measurements': vol.Any(None, [str]),
        'include_measurements': vol.Any(None, [str]),
    }

    _output_schema_dict = {
        'observables':  [iio.coerce_observable_dict],
        'exclude_measurements': vol.All(iio.ensurelist, [str]),
        'include_measurements': vol.All(iio.ensurelist, [str]),
    }

    def __init__(self, observables, *,
                 exclude_measurements=None,
                 include_measurements=None,
                 include_pseudo_measurements=False):
        """Initialize the instance.

        Parameters:
        - `observables`: list of observables (tuples or strings)
        - `exclude_measurements`: list of measurement names to exclude
        (default: none ar excluded)
        - `include_measurements`: list of measurement names to include
        (default: all are included)

        Only one of `exclude_measurements` or `include_measurements` should
        be specified. By default, all existing instances of
        `flavio.Measurements` are included as constraints (except if they
        carry 'Pseudo-measurement' in their name).
        """
        super().__init__()
        self.observables = observables
        self.exclude_measurements = exclude_measurements
        self.include_measurements = include_measurements
        self.include_pseudo_measurements = include_pseudo_measurements
        if exclude_measurements and include_measurements:
            raise ValueError("The options exclude_measurements and include_measurements must not be specified simultaneously")

        # check that observables are constrained
        _obs_measured = set()
        for m_name in self.get_measurements:
            m_obj = flavio.Measurement[m_name]
            _obs_measured.update(m_obj.all_parameters)
        missing_obs = set(observables) - set(_obs_measured).intersection(set(observables))
        assert missing_obs == set(), "No measurement found for the observables: " + str(missing_obs)
        self._warn_meas_corr()

    def _warn_meas_corr(self):
        """Warn the user if the fit contains multiple correlated measurements of
        an observable that is not included in the fit parameters, as this will
        lead to inconsistent results."""
        corr_with = {}
        # iterate over all measurements constraining at least one fit obs.
        for name in self.get_measurements:
            m = flavio.classes.Measurement[name]
            # iterate over all fit obs. constrained by this measurement
            for obs in set(self.observables) & set(m.all_parameters):
                # the constraint on this fit obs.
                constraint = m._parameters[obs][1]
                # find all the other obs. constrained by this constraint
                for c, p in m._constraints:
                    if c == constraint:
                        par = p
                        break
                for p in par:
                    # if the other obs. are not fit obs., append them to the list
                    if p not in self.observables:
                        if p not in corr_with:
                            corr_with[p] = [obs]
                        else:
                            corr_with[p].append(obs)
        # replace list by a Counter
        corr_with = {k: Counter(v) for k, v in corr_with.items() if v}
        # warn for all counts > 1
        for obs1, counter in corr_with.items():
            for obs2, count in counter.items():
                if count > 1:
                    warnings.warn(("{} of the measurements in the likelihood "
                                   "constrain both '{}' and '{}', but only the "
                                   "latter is included among the fit "
                                   "observables. This can lead to inconsistent "
                                   "results as the former is profiled over."
                                   ).format(count, obs1, obs2))
        return corr_with

    @property
    def get_measurements(self):
        """Return a list of all the measurements currently defined that
        constrain any of the fit observables."""
        all_measurements = []
        for m_name, m_obj in flavio.classes.Measurement.instances.items():
            if m_name.split(' ')[0] == 'Pseudo-measurement' and not self.include_pseudo_measurements:
                # skip pseudo measurements generated by FastFit instances
                continue
            if set(m_obj.all_parameters).isdisjoint(self.observables):
                # if set of all observables constrained by measurement is disjoint
                # with fit observables, do nothing
                continue
            else:
                # else, add measurement name to output list
                all_measurements.append(m_name)
        if self.exclude_measurements is None and self.include_measurements is None:
            return all_measurements
        elif self.exclude_measurements is not None:
            return list(set(all_measurements) - set(self.exclude_measurements))
        elif self.include_measurements is not None:
            return list(set(all_measurements) & set(self.include_measurements))

    def get_predictions_par(self, par_dict, wc_obj):
        """Compute the predictions for all observables as functions of
        a parameter dictionary `par_dict`and WilsonCoefficient instance
        `wc_obj`"""
        all_predictions = {}
        for observable in self.observables:
            obs = flavio.classes.Observable.argument_format(observable, 'dict')
            name = obs.pop('name')
            _inst = flavio.classes.Observable[name]
            all_predictions[observable] = _inst.prediction_par(par_dict, wc_obj, **obs)
        return all_predictions

    def log_likelihood_pred(self, pred_dict):
        """Return the logarithm of the likelihood function as a function of
        a dictionary of observable predictions `pred_dict`"""
        ll = 0.
        for measurement in self.get_measurements:
            m_obj = flavio.Measurement[measurement]
            m_obs = m_obj.all_parameters
            exclude_observables = set(m_obs) - set(self.observables)
            prob_dict = m_obj.get_logprobability_all(pred_dict, exclude_parameters=exclude_observables)
            ll += sum(prob_dict.values())
        return ll

    def log_likelihood_par(self, par_dict, wc_obj):
        """Return the logarithm of the likelihood function as a function of
        a parameter dictionary `par_dict` and WilsonCoefficient instance
        `wc_obj`"""
        predictions = self.get_predictions_par(par_dict, wc_obj)
        return self.log_likelihood_pred(predictions)


class ParameterLikelihood(iio.YAMLLoadable):
    """A `ParameterLikelihood` provides a likelihood function in terms of
    parameters.

    Methods:

    - `log_likelihood_par`: The likelihood as a function of the parameters
    - `get_central`: get an array with the parameters' central values
    - `get_random`: get an array with random values for the parameters

    Instances can be imported and exported from/to YAML using the `load`
    and `dump` methods.
    """
    _input_schema_dict = {
        'par_obj': vol.All([dict], iio.coerce_par_obj),
        'parameters': vol.Any(None, [str]),
    }

    _output_schema_dict = {
        'par_obj': iio.get_par_diff,
        'parameters': vol.Any(iio.ensurelist, [str]),
    }

    def __init__(self,
                 par_obj=flavio.default_parameters,
                 parameters=None):
        """Initialize the instance.

        Parameters:

        - `par_obj`: an instance of `ParameterConstraints` (defaults to
        `flavio.default_parameters`)
        - parameters: a list of parameters whose constraints should be taken
        into account in the likelihood.
        """
        self.par_obj = par_obj
        self.parameters = parameters
        self.parameters_central = self.par_obj.get_central_all()

    def log_likelihood_par(self, par_dict):
        """Return the prior probability for all parameters.

        Note that only the parameters in `self.parameters` will give a
        contribution to the likelihood."""
        exclude_parameters = list(set(self.par_obj._parameters.keys())-set(self.parameters))
        prob_dict = self.par_obj.get_logprobability_all(par_dict, exclude_parameters=exclude_parameters)
        return sum([p for obj, p in prob_dict.items()])

    @property
    def get_central(self):
        """Return a numpy array with the central values of all parameters."""
        return np.asarray([self.parameters_central[p] for p in self.parameters])

    @property
    def get_random(self):
        """Return a numpy array with random values for all parameters."""
        all_random = self.par_obj.get_random_all()
        return np.asarray([all_random[p] for p in self.parameters])


class Likelihood(iio.YAMLLoadable):
    """A `Likelihood` provides a likelihood function consisting of a
    contribution from experimental measurements and a contribution from
    parameters.

    Methods:

    - `log_prior_fit_parameters`: The parameter contribution to the
    log-likelihood
    - `log_likelihood_exp`: The experimental contribution to the
    log-likelihood
    - `log_likelihood`: The total log-likelihood that is the sum of the
    parameter and the experimental contribution

    Instances can be imported and exported from/to YAML using the `load`
    and `dump` methods.
    """

    _input_schema_dict = {
        'par_obj': vol.All([dict], iio.coerce_par_obj),
        'fit_parameters': vol.Any(iio.ensurelist, [str]),
        'observables':  vol.All([iio.coerce_observable_tuple], iio.list_deduplicate),
        'exclude_measurements': vol.Any(iio.ensurelist, [str]),
        'include_measurements': vol.Any(iio.ensurelist, [str]),
    }

    _output_schema_dict = {
        'par_obj': vol.All([dict], iio.coerce_par_obj),
        'fit_parameters': iio.get_par_diff,
        'observables':  [iio.coerce_observable_dict],
        'exclude_measurements': vol.Any(iio.ensurelist, [str]),
        'include_measurements': vol.Any(iio.ensurelist, [str]),
    }

    def __init__(self,
                 par_obj=flavio.default_parameters,
                 fit_parameters=None,
                 observables=None,
                 exclude_measurements=None,
                 include_measurements=None,
                 include_pseudo_measurements=False,
                 ):
        self.par_obj = par_obj
        self.parameters_central = self.par_obj.get_central_all()
        self.fit_parameters = fit_parameters or []
        self.observables = observables
        self.exclude_measurements = exclude_measurements
        self.include_measurements = include_measurements
        self.measurement_likelihood = MeasurementLikelihood(
            observables,
            exclude_measurements=exclude_measurements,
            include_measurements=include_measurements,
            include_pseudo_measurements=include_pseudo_measurements,)
        self.parameter_likelihood = ParameterLikelihood(
            par_obj=par_obj,
            parameters=fit_parameters)

    def log_prior_fit_parameters(self, par_dict):
        """Parameter contribution to the log-likelihood."""
        if not self.fit_parameters:
            return 0  # nothing to do
        return self.parameter_likelihood.log_likelihood_par(par_dict)

    def log_likelihood_exp(self, par_dict, wc_obj):
        """Experimental contribution to the log-likelihood."""
        return self.measurement_likelihood.log_likelihood_par(par_dict, wc_obj)

    def log_likelihood(self, par_dict, wc_obj):
        """Total log-likelihood.

        Parameters:
        - `par_dict`: a dictionary of parameter values
        - `wc_obj`: an instance of `WilsonCoefficients` or `wilson.Wilson`
        """
        return self.log_prior_fit_parameters(par_dict) + self.log_likelihood_exp(par_dict, wc_obj)


class SMCovariance(object):
    """Class to compute, save, and load a covariance matrix of SM
    predictions.

    Methods:

    - `compute`: Compute the covariance
    - `get`: Compute the covariance if necessary, otherwise return cached one
    - `save`: Save the covariance to a file
    - `load`: Load the covariance from a file
    - `load_dict`: Load the covariance from a dictionary
    """

    def __init__(self, observables, *,
                 vary_parameters='all', par_obj=None):
        """Initialize the class.

        Parameters:
        - `observables`: list of observables
        - `vary_parameters`: parameters to vary. Defaults to 'all'.
        - `par_obj`: instance of ParameterConstraints. Defaults to
        flavio.default_parameters.
        """
        self.observables = observables
        self.vary_parameters = vary_parameters
        self.par_obj = par_obj or flavio.default_parameters
        self._cov = None

    def compute(self, N, threads):
        """Compute the covariance for `N` random values, using `threads`
        CPU threads."""
        return flavio.sm_covariance(obs_list=self.observables,
                                    N=N,
                                    par_vary=self.vary_parameters,
                                    par_obj=self.par_obj,
                                    threads=threads)

    def get(self, N=100, threads=1, force=True):
        """Compute the covariance for `N` random values (default: 100),
        using `threads` CPU threads (default: 1).

        If `force` is False, return a cached version if it exists.
        """
        if self._cov is None or force:
            self._cov = self.compute(N=N, threads=threads)
        elif N != 100:
            warnings.warn("Argument N={} ignored ".format(N) + \
                          "as covariance has already been " + \
                          "computed. Recompute using `force=True`.")
        return self._cov

    def save(self, filename):
        """Save the SM covariance to a pickle file.

        The covariance must have been computed before using `get` or
        `compute`.
        """
        if self._cov is None:
            raise ValueError("Call `get` or `compute` first.")
        with open(filename, 'wb') as f:
            data = dict(covariance=self._cov,
                        observables=self.observables)
            pickle.dump(data, f)

    def load(self, filename):
        """Load the SM covariance from a pickle file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.load_dict(d=data)

    def load_dict(self, d):
        """Load the SM covariance from a dictionary.

        It must have the form `{'observables': [...], 'covariance': [[...]]}`
        where 'covariance' is a covariance matrix in the basis of observables
        given by 'observables' which must at least contain all the observables
        involved in the fit. Additional observables will be ignored; the
        ordering is arbitrary.
        """
        obs = d['observables']
        try:
            permutation = [obs.index(o) for o in self.observables]
        except ValueError:
            raise ValueError("Covariance matrix does not contain all necessary entries")
        assert len(permutation) == len(self.observables), \
            "Covariance matrix does not contain all necessary entries"
        if len(permutation) == 1:
            if d['covariance'].shape == ():
                self._cov = d['covariance']
            else:
                self._cov = d['covariance'][permutation][:,permutation][0,0]
        else:
            self._cov = d['covariance'][permutation][:,permutation]


class MeasurementCovariance(object):
    """Class to compute, save, and load a covariance matrix and the central
    values of experimental measurements.

    Methods:

    - `compute`: Compute the covariance
    - `get`: Compute the covariance if necessary, otherwise return cached one
    - `save`: Save the covariance to a file
    - `load`: Load the covariance from a file
    - `load_dict`: Load the covariance from a dictionary
    """

    def __init__(self, measurement_likelihood):
        """Initialize the class.

        Parameters:
        - `measurement_likelihood`: an instance of `MeasurementLikelihood`
        """
        self.measurement_likelihood = measurement_likelihood
        self._central_cov = None

    def compute(self, N):
        """Compute the covariance for `N` random values."""
        ml = self.measurement_likelihood
        means = []
        covariances = []
        for measurement in ml.get_measurements:
            m_obj = flavio.Measurement[measurement]
            # obs. included in the fit and constrained by this measurement
            our_obs = set(m_obj.all_parameters).intersection(ml.observables)
            # construct a dict. containing a vector of N random values for
            # each of these observables
            random_dict = m_obj.get_random_all(size=N)
            random_arr = np.zeros((len(ml.observables), N))
            for i, obs in enumerate(ml.observables):
                if obs in our_obs:
                    random_arr[i] = random_dict[obs]
            mean = np.mean(random_arr, axis=1)
            covariance = np.cov(random_arr)
            for i, obs in enumerate(ml.observables):
                if obs not in our_obs:
                    covariance[:,i] = 0
                    covariance[i, :] = 0
                    covariance[i, i] = np.inf
            means.append(mean)
            covariances.append(covariance)
        # if there is only a single measuement
        if len(means) == 1:
            return means[0], covariances[0]
        # if there are severeal measurements, perform a weighted average
        else:
            # covariances: [Sigma_1, Sigma_2, ...]
            # means: [x_1, x_2, ...]
            # weights_ [W_1, W_2, ...] where W_i = (Sigma_i)^(-1)
            # weighted covariance is  (W_1 + W_2 + ...)^(-1) = Sigma
            # weigted mean is  Sigma.(W_1.x_1 + W_2.x_2 + ...) = x
            if len(ml.observables) == 1:
                weights = np.array([1/c for c in covariances])
                weighted_covariance = 1/np.sum(weights, axis=0)
                weighted_mean = weighted_covariance * np.sum(
                                [np.dot(weights[i], means[i]) for i in range(len(means))])
            else:
                weights = [np.linalg.inv(c) for c in covariances]
                weighted_covariance = np.linalg.inv(np.sum(weights, axis=0))
                weighted_mean = np.dot(weighted_covariance, np.sum(
                                [np.dot(weights[i], means[i]) for i in range(len(means))],
                                axis=0))
            return weighted_mean, weighted_covariance

    def get(self, N=5000, force=True):
        """Compute the covariance for `N` random values (default: 5000).

        If `force` is False, return a cached version if it exists.
        """
        if self._central_cov is None or force:
            self._central_cov = self.compute(N=N)
        elif N != 5000:
            warnings.warn("Argument N={} ignored ".format(N) + \
                          "as experimental covariance has already been " + \
                          "computed. Recompute using `force=True`.")
        return self._central_cov

    def save(self, filename):
        """Save the experimental central values and the covariance to a pickle
        file.

        The central values and the covariance must have been computed before
        using `get` or `compute`."""
        if self._central_cov is None:
            raise ValueError("Call `get` or `compute` first.")
        with open(filename, 'wb') as f:
            data = dict(central=self._central_cov[0],
                        covariance=self._central_cov[1],
                        observables=self.measurement_likelihood.observables)
            pickle.dump(data, f)

    def load(self, filename):
        """Load the experimental central values and the covriance from a pickle
        file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.load_dict(d=data)

    def load_dict(self, d):
        """Load the the experimental central values and the covariance from a
        dictionary.

        It must have the form
        `{'observables': [...], 'central': [...], 'covariance': [[...]]}`
        where 'central' is a vector of central values and 'covariance' is a
        covariance matrix, both in the basis of observables given by
        'observables' which must at least contain all the observables
        involved in the fit. Additional observables will be ignored; the
        ordering is arbitrary."""
        ml = self.measurement_likelihood
        obs = d['observables']
        try:
            permutation = [obs.index(o) for o in ml.observables]
        except ValueError:
            raise ValueError("Covariance matrix does not contain all necessary entries")
        assert len(permutation) == len(ml.observables), \
            "Covariance matrix does not contain all necessary entries"
        if len(permutation) == 1:
            self._exp_central_covariance = (
                d['central'],
                d['covariance']
            )
        else:
            self._exp_central_covariance = (
                d['central'][permutation],
                d['covariance'][permutation][:, permutation],
            )


class FastLikelihood(NamedInstanceClass, iio.YAMLLoadable):
    """A variant (but not subclass) of `Likelihood` where some or all
    of the parameters have been "integrated out" and their theoretical
    uncertainties combined with the experimental uncertainties into
    a multivariate Gaussian "pseudo measurement".

    The pseuo measurement is generated by calling the method `make_measurement`.
    This is done by
    generating random samples of the nuisance parameters and evaluating all
    observables within the Standard Model many times (100 by default).
    Then, the covariance of all predictions is extracted. Similarly, a covariance
    matrix for all experimental measurements is determined. Both covariance
    matrices are added and the resulting multivariate Gaussian treated as a
    single measurement.

    This approach has the advantage that two-dimensional plots of the likelihood
    can be produced without the need for sampling or profiling the other
    dimensions. However, several strong assumptions go into this method, most
    importantly,

    - all uncertainties - experimental and theoretical - are treated as Gaussian
    - the theoretical uncertainties in the presence of new physics are assumed
      to be equal to the ones in the SM

    Methods:

    - `make_measurement`: Generate the pseudo measurement
    - `log_likelihood`: The log-likelihood function

    Important attributes/properties:

    - `likelihood`: the `Likelihood` instance based on the pseudo measurement
    - `full_measurement_likelihood`: the `MeasurementLikelihood` instance based
    on the original measurements
    - `sm_covariance`: the `SMCovariance` instance (that can be used to load
    and save the SM covariance matrix)
    - `exp_covariance`: the `MeasurementCovariance` instance (that can be used
    to load and save the experimental covariance matrix)

    Instances can be imported and exported from/to YAML using the `load`
    and `dump` methods.
    """

    _input_schema_dict = {
        'name': str,
        'par_obj': vol.All([dict], iio.coerce_par_obj),
        'fit_parameters': vol.Any(None, [str]),
        'nuisance_parameters': vol.Any(None, [str]),
        'observables':  vol.All([iio.coerce_observable_tuple], iio.list_deduplicate),
        'exclude_measurements': vol.Any(None, [str]),
        'include_measurements': vol.Any(None, [str]),
    }

    _output_schema_dict = {
        'name': str,
        'par_obj': iio.get_par_diff,
        'fit_parameters': vol.Any(iio.ensurelist, [str]),
        'nuisance_parameters': vol.Any(iio.ensurelist, [str]),
        'observables':  [iio.coerce_observable_dict],
        'exclude_measurements': vol.Any(iio.ensurelist, [str]),
        'include_measurements': vol.Any(iio.ensurelist, [str]),
    }

    def __init__(self, name,
                 par_obj=flavio.default_parameters,
                 fit_parameters=None,
                 nuisance_parameters='all',
                 observables=None,
                 exclude_measurements=None,
                 include_measurements=None,
                 ):
        self.par_obj = par_obj
        self.parameters_central = self.par_obj.get_central_all()
        self.fit_parameters = fit_parameters or []
        if nuisance_parameters == 'all':
            self.nuisance_parameters = self.par_obj.all_parameters
        else:
            self.nuisance_parameters = nuisance_parameters or []
        self.observables = observables
        self.exclude_measurements = exclude_measurements
        self.include_measurements = include_measurements
        self.full_measurement_likelihood = MeasurementLikelihood(
            self.observables,
            exclude_measurements=self.exclude_measurements,
            include_measurements=self.include_measurements)
        self.sm_covariance = SMCovariance(self.observables,
            vary_parameters=self.nuisance_parameters,
            par_obj=self.par_obj)
        self.exp_covariance = MeasurementCovariance(
            self.full_measurement_likelihood)
        self.pseudo_measurement = None
        self._likelihood = None
        NamedInstanceClass.__init__(self, name)

    def make_measurement(self, N=100, Nexp=5000, threads=1, force=False, force_exp=False):
        """Initialize the likelihood by producing a pseudo-measurement containing both
        experimental uncertainties as well as theory uncertainties stemming
        from nuisance parameters.

        Optional parameters:

        - `N`: number of random computations for the SM covariance (computing
          time is proportional to it; more means less random fluctuations.)
        - `Nexp`: number of random computations for the experimental covariance.
          This is much less expensive than the theory covariance, so a large
          number can be afforded (default: 5000).
        - `threads`: number of parallel threads for the SM
          covariance computation. Defaults to 1 (no parallelization).
        - `force`: if True, will recompute SM covariance even if it
          already has been computed. Defaults to False.
        - `force_exp`: if True, will recompute experimental central values and
          covariance even if they have already been computed. Defaults to False.
        """
        central_exp, cov_exp = self.exp_covariance.get(Nexp, force=force_exp)
        cov_sm = self.sm_covariance.get(N, force=force, threads=threads)
        covariance = cov_exp + cov_sm
        # add the Pseudo-measurement
        m = flavio.classes.Measurement('Pseudo-measurement for FastLikelihood instance: ' + self.name)
        if np.asarray(central_exp).ndim == 0 or len(central_exp) <= 1: # for a 1D (or 0D) array
            m.add_constraint(self.observables,
                    NormalDistribution(central_exp, np.sqrt(covariance)))
        else:
            m.add_constraint(self.observables,
                    MultivariateNormalDistribution(central_exp, covariance))
        self.pseudo_measurement = m
        self._likelihood = Likelihood(
            par_obj=self.par_obj,
            fit_parameters=self.fit_parameters,
            observables=self.observables,
            include_measurements=[m.name],  # only include our pseudo-meas.
            include_pseudo_measurements=True,  # force including the pseudo-meas.
            )

    @property
    def likelihood(self):
        if self._likelihood is None:
            raise ValueError("You need to call `make_measurement` first.")
        return self._likelihood

    def log_likelihood(self, par_dict, wc_obj):
        """Log-likelihood function.

        Parameters:
        - `par_dict`: a dictionary of parameter values
        - `wc_obj`: an instance of `WilsonCoefficients` or `wilson.Wilson`
        """
        return self.likelihood.log_likelihood_exp(par_dict, wc_obj)
