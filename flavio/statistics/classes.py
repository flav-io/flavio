import flavio
import numpy as np


class Fit(flavio.classes.NamedInstanceClass):
    """Base class for fits"""

    def __init__(self,
                 name,
                 constraints,
                 wc_obj,
                 fit_parameters,
                 nuisance_parameters,
                 fit_coefficients,
                 measurements,
                 exclude_observables=[],
                 input_scale=160.
                ):
        # some checks to make sure the input is sane
        for p in fit_parameters + nuisance_parameters:
            assert p in constraints._parameters.keys(), "Parameter " + p + " not found in Constraints"
        for m in measurements:
            try:
                flavio.classes.Measurement.get_instance(m)
            except:
                raise ValueError("Measurement " + m + " not found!")
        for obs in exclude_observables:
            try:
                if isinstance(obs, tuple):
                    flavio.classes.Observable.get_instance(obs[0])
                else:
                    flavio.classes.Observable.get_instance(obs)
            except:
                raise ValueError("Observable " + str(obs) + " not found!")
        intersect = set(fit_parameters).intersection(nuisance_parameters)
        assert intersect == set(), "Parameters appearing as fit_parameters and nuisance_parameters: " + str(intersect)
        # now that everything seems fine, we can call the init of the parent class
        for c in fit_coefficients :
            assert c in wc_obj.all_wc, "Wilson coefficient " + c + " not found"
        super().__init__(name)
        self.constraints = constraints
        self.parameters_central = self.constraints.get_central_all()
        self.wc_obj = wc_obj
        self.fit_parameters = fit_parameters
        self.nuisance_parameters = nuisance_parameters
        self.fit_coefficients = fit_coefficients
        self.input_scale = input_scale

    @property
    def get_central_fit_parameters(self):
        """Return a numpy array with the central values of all fit parameters."""
        return np.asarray([self.parameters_central[p] for p in self.fit_parameters])

    @property
    def get_random_fit_parameters(self):
        """Return a numpy array with random values for all fit parameters."""
        all_random = self.constraints.get_random_all()
        return np.asarray([all_random[p] for p in self.fit_parameters])

    @property
    def get_central_nuisance_parameters(self):
        """Return a numpy array with the central values of all nuisance parameters."""
        return np.asarray([self.parameters_central[p] for p in self.nuisance_parameters])

    @property
    def get_random_nuisance_parameters(self):
        """Return a numpy array with random values for all nuisance parameters."""
        all_random = self.constraints.get_random_all()
        return np.asarray([all_random[p] for p in self.nuisance_parameters])


class BayesianFit(Fit):
    """Bayesian fit class"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # * 2 for real and imaginary part
        self.dimension = len(self.fit_parameters) + len(self.nuisance_parameters) + len(self.fit_coefficients) * 2

    def array_to_dict(self, x):
        """Convert a 1D numpy array of floats to a dictionary of fit parameters,
        nuisance parameters, and Wilson coefficients."""
        d = {}
        n_fit_p = len(self.fit_parameters)
        n_nui_p = len(self.nuisance_parameters)
        n_wc = len(self.fit_coefficients) * 2  # * 2 for real and imaginary part
        d['fit_parameters'] = { p: x[i] for i, p in enumerate(self.fit_parameters) }
        d['nuisance_parameters'] = { p: x[i + n_fit_p] for i, p in enumerate(self.nuisance_parameters) }
        d['fit_coefficients'] = { p: x[2*i + n_fit_p + n_nui_p] + 1j*x[2*i+1 + n_fit_p + n_nui_p] for i, p in enumerate(self.fit_coefficients) }
        return d

    def dict_to_array(self, d):
        """Convert a dictionary of fit parameters,
        nuisance parameters, and Wilson coefficients to a 1D numpy array of
        floats."""
        n_fit_p = len(self.fit_parameters)
        n_nui_p = len(self.nuisance_parameters)
        n_wc = len(self.fit_coefficients) * 2  # * 2 for real and imaginary part
        arr = np.zeros(n_fit_p + n_nui_p + n_wc)
        arr[:n_fit_p] = [d['fit_parameters'][p] for p in self.fit_parameters]
        arr[n_fit_p:n_fit_p+n_nui_p] = [d['nuisance_parameters'][p] for p in self.nuisance_parameters]
        arr[n_fit_p+n_nui_p::2]   = [d['fit_coefficients'][c].real for c in self.fit_coefficients]
        arr[n_fit_p+n_nui_p+1::2] = [d['fit_coefficients'][c].imag for c in self.fit_coefficients]
        return arr

    @property
    def get_random(self):
        arr = np.zeros(self.dimension)
        n_fit_p = len(self.fit_parameters)
        n_nui_p = len(self.nuisance_parameters)
        arr[:n_fit_p] = self.get_random_fit_parameters
        arr[n_fit_p:n_fit_p+n_nui_p] = self.get_random_nuisance_parameters
        # TODO: random Wilson coefficients (priors?)
        return arr

    def log_likelihood(self, x):
        d = self.array_to_dict(x)
        par_dict = self.parameters_central.copy()
        par_dict.update(d['fit_parameters'])
        par_dict.update(d['nuisance_parameters'])
        prob_dict = self.constraints.get_logprobability_all(par_dict)
        return sum([p for obj, p in prob_dict.items()])
