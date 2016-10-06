"""A fit is a collection of observables and parameters that can be used to
perform statistical analyses within a particular statistical framework.

Fits are instances of descendants of the `Fit` class (which is not meant
to be used directly)."""

import flavio
import numpy as np
import copy

class Fit(flavio.NamedInstanceClass):
    """Base class for fits. Not meant to be used directly."""

    def __init__(self,
                 name,
                 par_obj,
                 fit_parameters,
                 nuisance_parameters,
                 observables,
                 fit_wc_names=[],
                 fit_wc_function=None,
                 fit_wc_priors=None,
                 input_scale=160.,
                 exclude_measurements=None,
                 include_measurements=None,
                ):
        # some checks to make sure the input is sane
        for p in fit_parameters + nuisance_parameters:
            # check that fit and nuisance parameters exist
            assert p in par_obj._parameters.keys(), "Parameter " + p + " not found in Constraints"
        for obs in observables:
            # check that observables exist
            try:
                if isinstance(obs, tuple):
                    flavio.classes.Observable.get_instance(obs[0])
                else:
                    flavio.classes.Observable.get_instance(obs)
            except:
                raise ValueError("Observable " + str(obs) + " not found!")
        if exclude_measurements is not None and include_measurements is not None:
            raise ValueError("The options exclude_measurements and include_measurements must not be specified simultaneously")
        # check that no parameter appears as fit *and* nuisance parameter
        intersect = set(fit_parameters).intersection(nuisance_parameters)
        assert intersect == set(), "Parameters appearing as fit_parameters and nuisance_parameters: " + str(intersect)
        # check that the Wilson coefficient function works
        if fit_wc_names: # if list of WC names not empty
            try:
                fit_wc_function(**{fit_wc_name: 1e-6 for fit_wc_name in fit_wc_names})
            except:
                raise ValueError("Error in calling the Wilson coefficient function")
        # now that everything seems fine, we can call the init of the parent class
        super().__init__(name)
        self.par_obj = par_obj
        self.parameters_central = self.par_obj.get_central_all()
        self.fit_parameters = fit_parameters
        self.nuisance_parameters = nuisance_parameters
        self.exclude_measurements = exclude_measurements
        self.include_measurements = include_measurements
        self.fit_wc_names = fit_wc_names
        self.fit_wc_function = fit_wc_function
        self.fit_wc_priors = fit_wc_priors
        self.observables = observables
        self.input_scale = input_scale

    @property
    def get_central_fit_parameters(self):
        """Return a numpy array with the central values of all fit parameters."""
        return np.asarray([self.parameters_central[p] for p in self.fit_parameters])

    @property
    def get_random_fit_parameters(self):
        """Return a numpy array with random values for all fit parameters."""
        all_random = self.par_obj.get_random_all()
        return np.asarray([all_random[p] for p in self.fit_parameters])

    @property
    def get_random_wilson_coeffs(self):
        """Return a numpy array with random values for all Wilson coefficients."""
        if self.fit_wc_priors is None:
            return None
        all_random = self.fit_wc_priors.get_random_all()
        return np.asarray([all_random[p] for p in self.fit_wc_names])

    @property
    def get_central_nuisance_parameters(self):
        """Return a numpy array with the central values of all nuisance parameters."""
        return np.asarray([self.parameters_central[p] for p in self.nuisance_parameters])

    @property
    def get_random_nuisance_parameters(self):
        """Return a numpy array with random values for all nuisance parameters."""
        all_random = self.par_obj.get_random_all()
        return np.asarray([all_random[p] for p in self.nuisance_parameters])

    @property
    def get_measurements(self):
        """Return a list of all the measurements currently defined that
        constrain any of the fit observables."""
        all_measurements = []
        for m_name, m_obj in flavio.classes.Measurement.instances.items():
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



class BayesianFit(Fit):
    """Bayesian fit class. Instances of this class can then be fed to samplers.

    Parameters
    ----------

    - `name`: a descriptive string name
    - `par_obj`: an instance of `ParameterConstraints`, e.g. `flavio.default_parameters`
    - `fit_parameters`: a list of string names of parameters of interest. The existing
      constraints on the parameter will be taken as prior.
    - `nuisance_parameters`: a list of string names of nuisance parameters. The existing
      constraints on the parameter will be taken as prior.
    - `observables`: a list of observable names to be included in the fit
    - `exclude_measurements`: optional; a list of measurement names *not* to be included in
    the fit. By default, all existing measurements are included.
    - `include_measurements`: optional; a list of measurement names to be included in
    the fit. By default, all existing measurements are included.
    - `fit_wc_names`: optional; a list of string names of arguments of the Wilson
      coefficient function below
    - `fit_wc_function`: optional; a function that has exactly the arguements listed
      in `fit_wc_names` and returns a dictionary that can be fed to the `set_initial`
      method of the Wilson coefficient class. Example: fit the real and imaginary
      parts of $C_{10}$ in $b\to s\mu^+\mu^-$.
    ```
    def fit_wc_function(Re_C10, Im_C10):
        return {'C10_bsmmumu': Re_C10 + 1j*Im_C10}
    ```
    - `input_scale`: input scale for the Wilson coeffficients. Defaults to 160.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dimension = len(self.fit_parameters) + len(self.nuisance_parameters) + len(self.fit_wc_names)

    def array_to_dict(self, x):
        """Convert a 1D numpy array of floats to a dictionary of fit parameters,
        nuisance parameters, and Wilson coefficients."""
        d = {}
        n_fit_p = len(self.fit_parameters)
        n_nui_p = len(self.nuisance_parameters)
        n_wc = len(self.fit_wc_names)
        d['fit_parameters'] = { p: x[i] for i, p in enumerate(self.fit_parameters) }
        d['nuisance_parameters'] = { p: x[i + n_fit_p] for i, p in enumerate(self.nuisance_parameters) }
        d['fit_wc'] = { p: x[i + n_fit_p + n_nui_p] for i, p in enumerate(self.fit_wc_names) }
        return d

    def dict_to_array(self, d):
        """Convert a dictionary of fit parameters,
        nuisance parameters, and Wilson coefficients to a 1D numpy array of
        floats."""
        n_fit_p = len(self.fit_parameters)
        n_nui_p = len(self.nuisance_parameters)
        n_wc = len(self.fit_wc_names)
        arr = np.zeros(n_fit_p + n_nui_p + n_wc)
        arr[:n_fit_p] = [d['fit_parameters'][p] for p in self.fit_parameters]
        arr[n_fit_p:n_fit_p+n_nui_p] = [d['nuisance_parameters'][p] for p in self.nuisance_parameters]
        arr[n_fit_p+n_nui_p:]   = [d['fit_wc'][c] for c in self.fit_wc_names]
        return arr

    @property
    def get_random(self):
        """Get an array with random values for all the fit and nuisance
        parameters"""
        arr = np.zeros(self.dimension)
        n_fit_p = len(self.fit_parameters)
        n_nui_p = len(self.nuisance_parameters)
        arr[:n_fit_p] = self.get_random_fit_parameters
        arr[n_fit_p:n_fit_p+n_nui_p] = self.get_random_nuisance_parameters
        arr[n_fit_p+n_nui_p:] = self.get_random_wilson_coeffs
        return arr

    def get_par_dict(self, x):
        """Get a dictionary of fit and nuisance parameters from an input array"""
        d = self.array_to_dict(x)
        par_dict = self.parameters_central.copy()
        par_dict.update(d['fit_parameters'])
        par_dict.update(d['nuisance_parameters'])
        return par_dict

    def get_wc_obj(self, x):
        wc_obj = flavio.WilsonCoefficients()
        # if there are no WCs to be fitted, return the SM WCs
        if not self.fit_wc_names:
            return wc_obj
        d = self.array_to_dict(x)
        wc_obj.set_initial(self.fit_wc_function(**d['fit_wc']), self.input_scale)
        return wc_obj

    def log_prior_parameters(self, x):
        """Return the prior probability for all fit and nuisance parameters
        given an input array"""
        par_dict = self.get_par_dict(x)
        exclude_parameters = list(set(par_dict.keys())-set(self.fit_parameters)-set(self.nuisance_parameters))
        prob_dict = self.par_obj.get_logprobability_all(par_dict, exclude_parameters=exclude_parameters)
        return sum([p for obj, p in prob_dict.items()])

    def log_prior_wilson_coeffs(self, x):
        """Return the prior probability for all Wilson coefficients
        given an input array"""
        if self.fit_wc_priors is None:
            return 0
        wc_dict = self.array_to_dict(x)['fit_wc']
        prob_dict = self.fit_wc_priors.get_logprobability_all(wc_dict)
        return sum([p for obj, p in prob_dict.items()])

    def get_predictions(self, x):
        """Get a dictionary with predictions for all observables given an input
        array"""
        par_dict = self.get_par_dict(x)
        wc_obj = self.get_wc_obj(x)
        all_predictions = {}
        for observable in self.observables:
            if isinstance(observable, tuple):
                obs_name = observable[0]
                _inst = flavio.classes.Observable.get_instance(obs_name)
                all_predictions[observable] = _inst.prediction_par(par_dict, wc_obj, *observable[1:])
            else:
                _inst = flavio.classes.Observable.get_instance(observable)
                all_predictions[observable] = _inst.prediction_par(par_dict, wc_obj)
        return all_predictions

    def log_likelihood(self, x):
        """Return the logarithm of the likelihood function (not including the
        prior)"""
        predictions = self.get_predictions(x)
        ll = 0.
        for measurement in self.get_measurements:
            m_obj = flavio.Measurement.get_instance(measurement)
            m_obs = m_obj.all_parameters
            exclude_observables = set(m_obs) - set(self.observables)
            prob_dict = m_obj.get_logprobability_all(predictions, exclude_parameters=exclude_observables)
            ll += sum(prob_dict.values())
        return ll

    def log_target(self, x):
        """Return the logarithm of the likelihood times prior probability"""
        return self.log_likelihood(x) + self.log_prior_parameters(x) + self.log_prior_wilson_coeffs(x)


class FastFit(BayesianFit):
    """A subclass of `BayesianFit` that is meant to produce fast likelihood
    contour plots.

    Calling the method `make_measurement`, a pseudo-measurement is generated
    that combines the actual experimental measurements with the theoretical
    uncertainties stemming from the nuisance parameters. This is done by
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
      to be similar to the ones in the SM
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.measurements = None


    # a method to get the mean and covariance of all measurements of all
    # observables of interest
    def _get_central_covariance_experiment(self, N=100):
        random_dict = {}
        for obs in self.observables:
            # intialize empty lists
            random_dict[obs] = []
        for measurement in self.get_measurements:
            if measurement.split(' ')[0] == 'Pseudo-measurement':
                continue
            m_obj = flavio.Measurement.get_instance(measurement)
            for i in range(N):
                m_random = m_obj.get_random_all()
                m_obs = m_obj.all_parameters
                our_obs = set(m_obs).intersection(self.observables)
                for obs in our_obs:
                    random_dict[obs].append(m_random[obs])
        random_arr = np.zeros((len(self.observables), N))
        for i, obs in enumerate(self.observables):
            n = len(random_dict[obs])
            random_arr[i] = random_dict[obs][::n//N]
        return np.mean(random_arr, axis=1), np.cov(random_arr)

    # a method to get the covariance of the SM prediction of all observables
    # of interest
    def _get_covariance_sm(self, N=100):
        par_central = self.par_obj.get_central_all()
        def random_nuisance_dict():
            arr = self.get_random_nuisance_parameters
            nuis_dict = {par: arr[i] for i, par in enumerate(self.nuisance_parameters)}
            par = par_central.copy()
            par.update(nuis_dict)
            return par
        par_random = [random_nuisance_dict() for i in range(N)]

        pred_arr = np.zeros((len(self.observables), N))
        wc_sm = flavio.WilsonCoefficients()
        for i, observable in enumerate(self.observables):
            if isinstance(observable, tuple):
                obs_name = observable[0]
                _inst = flavio.classes.Observable.get_instance(obs_name)
                pred_arr[i] = np.array([_inst.prediction_par(par, wc_sm, *observable[1:])
                                        for par in par_random])
            else:
                _inst = flavio.classes.Observable.get_instance(observable)
                pred_arr[i] = np.array([_inst.prediction_par(par, wc_sm)
                                        for par in par_random])
        return np.cov(pred_arr)

    def make_measurement(self, N=100, Nexp=1000):
        """Initialize the fit by producing a pseudo-measurement containing both
        experimental uncertainties as well as theory uncertainties stemming
        from nuisance parameters."""
        central_exp, cov_exp = self._get_central_covariance_experiment(Nexp)
        cov_sm = self._get_covariance_sm(N)
        covariance = cov_exp + cov_sm
        # add the Pseudo-measurement
        m = flavio.classes.Measurement('Pseudo-measurement for FastFit instance: ' + self.name)
        if len(central_exp) == 1:
            m.add_constraint(self.observables,
                    flavio.statistics.probability.NormalDistribution(central_exp, np.sqrt(covariance)))
        else:
            m.add_constraint(self.observables,
                    flavio.statistics.probability.MultivariateNormalDistribution(central_exp, covariance))

    def array_to_dict(self, x):
        """Convert a 1D numpy array of floats to a dictionary of fit parameters,
        nuisance parameters, and Wilson coefficients."""
        d = {}
        n_fit_p = len(self.fit_parameters)
        n_wc = len(self.fit_wc_names)
        d['fit_parameters'] = { p: x[i] for i, p in enumerate(self.fit_parameters) }
        d['fit_wc'] = { p: x[i + n_fit_p] for i, p in enumerate(self.fit_wc_names) }
        return d

    def dict_to_array(self, d):
        """Convert a dictionary of fit parameters and Wilson coefficients to a
        1D numpy array of floats."""
        n_fit_p = len(self.fit_parameters)
        n_wc = len(self.fit_wc_names)
        arr = np.zeros(n_fit_p + n_nui_p + n_wc)
        arr[:n_fit_p] = [d['fit_parameters'][p] for p in self.fit_parameters]
        arr[n_fit_p:]   = [d['fit_wc'][c] for c in self.fit_wc_names]
        return arr

    def get_par_dict(self, x):
        d = self.array_to_dict(x)
        par_dict = self.parameters_central.copy()
        par_dict.update(d['fit_parameters'])
        return par_dict

    def log_likelihood(self, x):
        """Return the logarithm of the likelihood. Note that there is no prior
        probability for nuisance parameters, which have been integrated out.
        Priors for fit parameters are ignored."""
        predictions = self.get_predictions(x)
        m_obj = flavio.Measurement.get_instance('Pseudo-measurement for FastFit instance: ' + self.name)
        m_obs = m_obj.all_parameters
        prob_dict = m_obj.get_logprobability_all(predictions)
        ll = sum(prob_dict.values())
        return ll
