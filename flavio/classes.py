"""Base classes for `flavio`"""


import numpy as np
from .config import config
from collections import OrderedDict, defaultdict
import copy
import flavio
from flavio._parse_errors import constraints_from_string, \
    convolve_distributions, dict2dist
from flavio.statistics.probability import string_to_class
import warnings
import yaml
import inspect
import urllib.parse
import re
import pkgutil


class NamedInstanceMetaclass(type):
    # this is just needed to implement the getitem method on NamedInstanceClass
    # to allow the syntax MyClass['instancename'] as shorthand for
    # MyClass.get_instance('instancename'); same for
    # del MyClass['instancename'] instead of MyClass.del_instance('instancename')
    def __getitem__(cls, item):
        return cls.get_instance(item)

    def __delitem__(cls, item):
        return cls.del_instance(item)


class NamedInstanceClass(object, metaclass=NamedInstanceMetaclass):
    """Base class for classes that have named instances that can be accessed
    by their name.

    Parameters
    ----------
     - name: string

    Methods
    -------
     - del_instance(name)
         Delete an instance
     - get_instance(name)
         Get an instance
     - set_description(description)
         Set the description
    """

    def __init__(self, name):
        if not hasattr(self.__class__, 'instances'):
            self.__class__.instances = OrderedDict()
        self.__class__.instances[name] = self
        self.name = name
        self.description = ''

    @classmethod
    def get_instance(cls, name):
        return cls.instances[name]

    @classmethod
    def del_instance(cls, name):
        del cls.instances[name]

    @classmethod
    def clear_all(cls):
        """Delete all instances."""
        cls.instances = OrderedDict()

    @classmethod
    def find(cls, regex):
        """Find all instance names matching the regular expression `regex`."""
        rc = re.compile(regex)
        return list(filter(rc.search, cls.instances))

    def set_description(self, description):
        self.description = description


class Parameter(NamedInstanceClass):
    """This class holds parameters (e.g. masses and lifetimes). It requires a
    name string and also allows to set a LaTeX name and description as
    attributes. Note that numerical values for the Parameters are not attributes
    of the Parameter class.

    Parameters
    ----------
     - name: string

    Attributes
    ----------
     - tex: string
     - description: string
    """

    def __init__(self, name):
        super().__init__(name)
        self.tex = ''

    def __repr__(self):
        return "Parameter('{}')".format(self.name)

    def _repr_markdown_(self):
        md = "### Parameter `{}`\n\n".format(self.name)
        if self.tex:
            md += "Parameter: {}\n\n".format(self.tex)
        if self.description:
            md += "Description: {}\n\n".format(self.description)
        return md


class Constraints(object):
    """Constraints are collections of probability distributions associated
    to objects like parameters or measurements. This is the base class of
    ParameterConstraints (that holds the numerical values and uncertainties
    of all the parameters) and Measurements (that holds the numerical values
    and uncertainties of all the experimental measurements.)

    Since this class is not meant for direct use, see these child classes for
    documentation.
    """

    def __init__(self):
            # Here we have two data structures. _constraints has the form
            # [ (<constraint1>, [parameter1, parameter2, ...]), (<constraint2>, ...) ]
            # where the <constraint>s are instances of ProbabilityDistribution
            # and the parameters string names, while _parameters has the form
            # { parameter1: (num1, <constraint1>)} where num1 is 0 for a
            # univariate constraint and otherwise gives the position of
            # parameter1 in the multivariate vector.
            # In summary, having this list and dictionary allow a bijective mapping between
            # constraints and parameters.
            # Note that one constraint can apply to multiple parameters (e.g.
            # in case of correlated uncertainties), but a parameter can only
            # have a single constraint (changed in v0.16!).
        self._constraints = []
        self._parameters = OrderedDict()

    @property
    def all_parameters(self):
        """Returns a list of all parameters/observables constrained."""
        return list(self._parameters.keys())

    def add_constraint(self, parameters, constraint):
        """Set the constraint on one or several parameters/observables.

        `constraint` must be an instance of a child of ProbabilityDistribution.

        Note that if there already exists a constraint, it will be removed."""
        for num, parameter in enumerate(parameters):
            # remove constraint if there is one
            if parameter in self._parameters:
                self.remove_constraint(parameter)
        # populate the dictionaries defined in __init__
            self._parameters[parameter] = (num, constraint)
        self._constraints.append((constraint, parameters))

    def set_constraint(self, parameter, constraint_string=None,
                                        constraint_dict=None):
        r"""Set the constraint on a parameter/observable by specifying a string
        or a dictionary. If several constraints (e.g. several types of
        uncertainty) are given, the total constraint will be the convolution
        of the individual distributions. Existing constraints will be removed.

        Arguments:

        - parameter: parameter string (or tuple)
        - constraint_string: string specifying the constraint that can be e.g.
          of the form `'1.55(3)(1)'` or `'4.0±0.1'`.
        - constraint_dict: dictionary or list of several dictionaries of the
          form `{'distribution': 'distribution_name', 'arg1': val1, ...}`, where
          'distribution_name' is a string name associated to each probability
          distribution (see `flavio.statistics.probability.class_from_string`)
          and `'arg1'`, `val1` are argument/value pairs of the arguments of
          the distribution class's constructor (e.g.`central_value`,
          `standard_deviation` for a normal distribution).

        `constraint_string` and `constraint_dict` must not be present
        simultaneously.
        """
        if constraint_string is not None and constraint_dict is not None:
            raise ValueError("constraint_string and constraint_dict cannot"
                             " be used at the same time.")
        if constraint_string is not None:
            pds = constraints_from_string(constraint_string)
        elif constraint_dict is not None:
            pds = dict2dist(constraint_dict)
        else:
            raise TypeError("Either constraint_string or constraint_dict have"
                            " to be specified.")
        combined_pd = convolve_distributions(pds)
        self.add_constraint([parameter], combined_pd)

    def remove_constraint(self, parameter):
        """Remove existing constraint on a parameter."""
        self._parameters.pop(parameter, None)

    def remove_constraints(self, parameter):
        warnings.warn("This function was renamed to `remove_constraint` "
                      "in v0.16 and will be removed in the future.",
                      DeprecationWarning)
        self.remove_constraint(parameter)

    def get_central(self, parameter):
        """Get the central value of a parameter"""
        if parameter not in self._parameters.keys():
            raise ValueError('No constraints applied to parameter/observable ' + parameter)
        else:
            num, constraint = self._parameters[parameter]
            cv = constraint.central_value
            try:
                cv = float(cv)
            except (TypeError, ValueError):
                # return the num-th entry of the central value vector
                return cv[num]
            else:
                if num == 0:
                    return cv
                else:
                    raise ValueError("Something went wrong when getting the central value of {}".format(parameter))

    def get_central_all(self):
        """Get central values of all constrained parameters."""
        return {parameter: self.get_central(parameter) for parameter in self._parameters.keys()}

    def get_random_all(self, size=None):
        """Get random values for all constrained parameters where they are
        distributed according to the probability distribution applied.

        If `size` is not None, the dictionary values will be arrays with length
        `size` rather than numbers."""
        # first, generate random values for every single one of the constraints
        random_constraints = {constraint: constraint.get_random(size=size)
                              for constraint, _ in self._constraints}
        random_dict = {}
        # now, iterate over the parameters
        for parameter, constraints in self._parameters.items():
            num, constraint = constraints
            carr = random_constraints[constraint]
            if size is None and num == 0 and np.isscalar(carr):
                random_dict[parameter] = carr
            elif size is None and num == 0 and carr.shape == tuple():
                random_dict[parameter] = carr
            elif size is None:
                random_dict[parameter] = carr[num]
            elif carr.shape == (size,) and num == 0:
                random_dict[parameter] = carr
            elif carr.ndim == 2 and carr.shape[0] == size:
                random_dict[parameter] = carr[:, num]
            else:
                raise ValueError("Unexpected error in get_random_all")
        return random_dict

    def get_1d_errors(self, N=1000):
        warnings.warn("This function was renamed to `get_1d_errors_random` "
                      "in v0.16 and will be removed in the future. ",
                      DeprecationWarning)
        self.get_1d_errors_random(N)

    def get_1d_errors_random(self, N=1000):
        """Get the Gaussian standard deviation for every parameter/observable
        obtained by generating N random values."""
        random_dict_list = [self.get_random_all() for i in range(N)]
        interval_dict = {}
        for k in random_dict_list[0].keys():
            arr = np.array([r[k] for r in random_dict_list])
            interval_dict[k] = np.std(arr)
        return interval_dict

    def get_1d_errors_rightleft(self):
        r"""Get the left and right error for every parameter/observable
        defined such that it contains 68% probability on each side of the
        central value."""
        errors_left = [constraint.error_left for constraint, _ in self._constraints]
        errors_right = [constraint.error_right for constraint, _ in self._constraints]
        error_dict = {}
        # now, iterate over the parameters
        for parameter, constraints in self._parameters.items():
            num, constraint = constraints
            idx = ([constraint for constraint, _ in self._constraints]).index(constraint)
            error_dict[parameter] = (np.ravel([errors_right[idx]])[num],
                                     np.ravel([errors_left[idx]])[num])
        return error_dict

    def get_logprobability_single(self, parameter, value, delta=False):
        """Return a dictionary with the logarithm of the probability for each
        constraint/probability distribution for a given value of a
        single parameter.
        """
        num, constraint = self._parameters[parameter]
        parameters = OrderedDict(self._constraints)[constraint]
        if len(parameters) == 1:
            if not delta:
                return constraint.logpdf(value)
            else:
                return constraint.delta_logpdf(value)
        else:
            # for multivariate distributions
            exclude = tuple(i for i, p in enumerate(parameters)
                            if p != parameter)
            if not delta:
                return constraint.logpdf([value], exclude=exclude)
            else:
                return constraint.delta_logpdf([value], exclude=exclude)

    def get_logprobability_all(self, par_dict, exclude_parameters=[], delta=False):
        """Return a dictionary with the logarithm of the probability for each
        constraint/probability distribution.

        Inputs
        ------
        - par_dict
          A dictionary of the form {parameter: value, ...} where parameter
          is a string and value a float.
        - exclude_parameters (optional)
          An iterable of strings (default: empty) that specifies parameters
          that should be ignored. Univariate constraints on this parameter
          will be skipped, while for multivariate normally distributed
          constraints, the parameter will be removed from the covariance.
        """
        prob_dict = {}
        for constraint, parameters in self._constraints:
            # list of constrained parameters except the excluded ones
            p_cons = [p for p in parameters
                      if (p not in exclude_parameters
                      and (parameters.index(p), constraint) == self._parameters.get(p, None))]
            x = [par_dict[p] for p in p_cons]
            if not x:
                # nothing to constrain
                continue
            if len(parameters) == 1:
                # 1D constraints should have a scalar, not a length-1 array
                if not delta:
                    prob_dict[constraint] = constraint.logpdf(x[0])
                else:
                    prob_dict[constraint] = constraint.delta_logpdf(x[0])
            else:
                # for multivariate distributions
                if len(x) == len(parameters):
                    # no parameter has been excluded
                    exclude = None
                else:
                    exclude = tuple(i for i, p in enumerate(parameters)
                                    if p not in p_cons)
                if not delta:
                    prob_dict[constraint] = constraint.logpdf(x, exclude=exclude)
                else:
                    prob_dict[constraint] = constraint.delta_logpdf(x, exclude=exclude)
        return prob_dict

    def copy(self):
        # this is to have a .copy() method like for a dictionary
        return copy.deepcopy(self)

    def get_yaml(self, *args, **kwargs):
        """Get a YAML string representation of all constraints.

        The optional parameter `pname` allows to customize the name of the key
        containing the parameter list of each constraint (e.g. 'parameters',
        'observables').
        """
        return yaml.dump(self.get_yaml_dict(*args, **kwargs))

    def get_yaml_dict(self, pname='parameters'):
        """Get an ordered dictionary representation of all constraints that can
        be dumped as YAML string.

        The optional parameter `pname` allows to customize the name of the key
        containing the parameter list of each constraint (e.g. 'parameters',
        'observables').
        """
        data = []
        for constraint, parameters in self._constraints:
            d = OrderedDict()
            d[pname] = [list(p) if isinstance(p, tuple) else p for p in parameters]
            d['values'] = constraint.get_dict(distribution=True,
                                              iterate=True, arraytolist=True)
            data.append(d)
        args = inspect.signature(self.__class__).parameters.keys()
        meta = {k: v for k, v in self.__dict__.items()
                if k[0] != '_' and v != '' and k not in args}
        if not args and not meta:
            return data
        else:
            datameta = OrderedDict()
            if args:
                datameta['arguments'] = {arg: self.__dict__[arg] for arg in args}
            if meta:
                datameta['metadata'] = meta
            datameta['constraints'] = data
            return datameta

    @classmethod
    def from_yaml(cls, stream, *args, **kwargs):
        """Class method: load constraint from a YAML string or stream."""
        data = yaml.safe_load(stream)
        return cls.from_yaml_dict(data, *args, **kwargs)

    @classmethod
    def from_yaml_dict(cls, data, pname='parameters', instance=None, *args, **kwargs):
        """Class method: load constraint from a dictionary or list of dicts.

        If it is a dictionary, it should have the form:

        ```{
        'metadata': {...},  # optional, do set attributes of the instance
        'arguments': {...},  # optional, to specify keyword arguments for instantiation,
        'constraints': [...],  # required, the list of constraints
        }

        Alternatively, the list of constraints can be directly given.
        This list should have elements in one of the two possible forms:

        1. Dictionary as returned by `Probability.get_dict`:
        ```{
        pname: [...],  # required, list of constrained parameters
        'values': {
            'distribution': '...',  # required, string identifying ProbabilityDistribution, e.g. 'normal'
            '...': '...',  # required, any arguments for the instantiation of the ProbabilityDistribution
            }
        }
        ```

        2. String representing one or several (to be convolved) constraints:
        ```{
        'my_parameter': '1.0 ± 0.2 ± 0.1 e-3'
        }
        """
        if isinstance(data, dict):
            constraints = data['constraints']
            meta = data.get('metadata', {})
            arguments = data['arguments']
            kwargs.update(arguments)
            inst = instance or cls(*args, **kwargs)
            for m in meta:
                inst.__dict__[m] = meta[m]
        else:
            inst = instance or cls(*args, **kwargs)
            constraints = data.copy()
        for c in constraints:
            if pname not in c:
                if 'values' not in c and len(c) == 1:
                    # this means we probably have a constraint of the
                    # form parameter: constraint_string
                    for k, v in c.items():  # this loop runs only once
                        inst.set_constraint(k, v)
                        break  # just to be sure
                    continue
                else:
                    # in this case something is clearly wrong. Mabye the
                    # wrong "pname" was used.
                    raise ValueError('Key ' + pname + ' not found. '
                                     'Please check the `pname` argument.')
            else:
                parameters = [tuple(p) if isinstance(p, list) else p for p in c[pname]]
                pds = dict2dist(c['values'])
                combined_pd = convolve_distributions(pds)
                inst.add_constraint(parameters, combined_pd)
        return inst


class ParameterConstraints(Constraints):
    """Trivial subclass of `Constraints` that is meant for constraints on
    theory parameters represented by instances of the `Parameter` class.
    """

    def __init__(self):
        super().__init__()

    def read_default(self):
        """Reset the instance and read the default parameters. Data is read
        - from 'data/parameters_metadata.yml'
        - from 'data/parameters_uncorrelated.yml'
        - from 'data/parameters_correlated.yml'
        - from the default PDG data file
        - for B->V form factors
        - for B->P form factors
        - for Lambdab->Lambda form factors
        """
        # import functions to read parameters
        from flavio.parameters import (
            _read_yaml_object_metadata,
            _read_yaml_object_values,
            _read_yaml_object_values_correlated,
            read_pdg
        )

        # reset the instance
        self.__init__()

        # Read the parameter metadata from the default YAML data file
        _read_yaml_object_metadata(pkgutil.get_data('flavio', 'data/parameters_metadata.yml'), self)

        # Read the uncorrelated parameter values from the default YAML data file
        _read_yaml_object_values(pkgutil.get_data('flavio', 'data/parameters_uncorrelated.yml'), self)

        # Read the correlated parameter values from the default YAML data file
        _read_yaml_object_values_correlated(pkgutil.get_data('flavio', 'data/parameters_correlated.yml'), self)

        # Read the parameters from the default PDG data file
        read_pdg(2018, self)

        # Read default parameters for B->V form factors
        flavio.physics.bdecays.formfactors.b_v.bsz_parameters.bsz_load('v2', 'LCSR', ('B->omega', 'B->rho'), self)
        flavio.physics.bdecays.formfactors.b_v.bsz_parameters.bsz_load('v2', 'LCSR-Lattice', ('B->K*', 'Bs->phi', 'Bs->K*'), self)

        # Read default parameters for B->P form factors
        flavio.physics.bdecays.formfactors.b_p.bsz_parameters.gkvd_load('v1', 'LCSR-Lattice', ('B->K', 'B->pi'), self)

        # Read default parameters for Lambdab->Lambda form factors
        flavio.physics.bdecays.formfactors.lambdab_12.lattice_parameters.lattice_load_ho(self)


class WilsonCoefficientPriors(Constraints):
    """
    """

    def __init__(self):
        super().__init__()


def tree():
    """Tree data structure.

    See https://gist.github.com/hrldcpr/2012250"""
    return defaultdict(tree)


def dicts(t):
    """Turn tree into nested dict"""
    return {k: dicts(t[k]) for k in t}


class Observable(NamedInstanceClass):
    """An Observable is something that can be measured experimentally and
    predicted theoretically."""

    def __init__(self, name, arguments=None):
        super().__init__(name)
        if not hasattr(self.__class__, 'taxonomy'):
            self.__class__.taxonomy = tree()
        self.arguments = arguments
        self.prediction = None
        self.tex = ''

    def __repr__(self):
        return "Observable('{}', arguments={})".format(self.name, self.arguments)

    def _repr_markdown_(self):
        md = "### Observable `{}`\n\n".format(self.name)
        if self.tex:
            md += "Observable: {}\n\n".format(self.tex)
        if self.description:
            md += "Description: {}\n\n".format(self.description)
        if self.arguments is not None:
            md += "Arguments: "
            md += ','.join(["`{}`".format(a) for a in self.arguments])
            md += "\n\n"
        if self.prediction is not None:
            f = self.prediction.function
            from IPython.lib import pretty
            md += "Theory prediction: `{}`".format(pretty.pretty(f))
        return md

    @classmethod
    def argument_format(cls, obs, format='tuple'):
        """Class method: takes as input an observable name and numerical values
        for the arguements (if any) and returns as output the same in a specific
        form as specified by `format`: 'tuple' (default), 'list', or 'dict'.

        Example inputs:
        - ('dBR/dq2(B0->Denu)', 1)
        - {'name': 'dBR/dq2(B0->Denu)', 'q2': 1}

        Output:
        tuple: ('dBR/dq2(B0->Denu)', 1)
        list: ('dBR/dq2(B0->Denu)', 1)
        dict: {'name': 'dBR/dq2(B0->Denu)', 'q2': 1}

        For a string input for observables that don't have arguments:
        - 'eps_K'

        Output:
        tuple: 'eps_K'
        list: 'eps_K'
        dict: {'name': 'eps_K'}
        """
        if isinstance(obs, str):
            if cls[obs].arguments is not None:
                raise ValueError("Arguments missing for {}".format(obs))
            if format == 'dict':
                return {'name': obs}
            else:
                return obs
        elif isinstance(obs, (tuple, list)):
            args = cls[obs[0]].arguments
            if args is None or len(args) != len(obs) - 1:
                raise ValueError("Wrong number of arguments for {}".format(obs[0]))
            t = tuple(obs)
            d = {'name': obs[0]}
            for i, a in enumerate(args):
                d[a] = obs[i + 1]
        elif isinstance(obs, dict):
            args = cls[obs['name']].arguments
            if args is None:
                t = obs['name']
            else:
                t = tuple([obs['name']] + [obs[a] for a in args])
            d = obs
        if format == 'tuple':
            return t
        elif format == 'list':
            return list(t)
        elif format == 'dict':
            return d

    def set_prediction(self, prediction):
        self.prediction = prediction

    def prediction_central(self, constraints_obj, wc_obj, *args, **kwargs):
        return self.prediction.get_central(constraints_obj, wc_obj, *args, **kwargs)

    def prediction_par(self, par_dict, wc_obj, *args, **kwargs):
        return self.prediction.get_par(par_dict, wc_obj, *args, **kwargs)

    def add_taxonomy(self, taxonomy_string):
        """Add a metadata taxonomy for the observable.

        `taxonomy_string` has to be a string of the form
        'Category :: Subcategory :: Subsubcategory'
        etc. LaTeX code is allowed. One observable can also have multiple
        taxonomies (e.g. 'Animal :: Cat' and 'Pet :: Favourite Pet')"""
        taxonomy_list = taxonomy_string.split(' :: ') + [self.name]
        t = self.__class__.taxonomy
        for node in taxonomy_list:
            t = t[node]

    @classmethod
    def taxonomy_dict(cls):
        """Return the hierarchical metadata taxonomy as a nested dictionary."""
        return dicts(cls.taxonomy)

    @classmethod
    def from_function(cls, name, observables, function):
        """Instantiate an observable object and the corresponding Prediction
        object for an observable that is defined as a mathematical function
        of two or more existing observables with existing predictions.

        Parameters:
        -----------

        - name: string name of the new observable
        - observables: list of string names of the observables to be combined
        - function: function of the observables. The number of arguments must
          match the number of observables

        Example:
        --------

        For two existing observables 'my_obs_1' and 'my_obs_2', a new observable
        that is defined as the difference between the two can be defined as

        ```
        Observable.from_function('my_obs_1_2_diff',
                                 ['my_obs_1', 'my_obs_2'],
                                 lambda x, y: x - y)
        ```
        """
        for observable in observables:
            try:
                Observable[observable]
            except KeyError:
                raise ValueError("The observable " + observable + " does not exist")
            assert Observable[observable].arguments == Observable[observables[0]].arguments, \
                "Only observables depending on the same arguments can be combined"
            assert Observable[observable].prediction is not None, \
                "The observable {} does not have a prediction yet".format(observable)
        obs_obj = cls(name, arguments=Observable[observables[0]].arguments)
        pfcts = [Observable[observable].prediction.function
                 for observable in observables]
        def pfct(*args, **kwargs):
            return function(*[f(*args, **kwargs) for f in pfcts])
        Prediction(name, pfct)
        return obs_obj

    def get_measurements(self):
        r"""Return the names of measurements that constrain the observable."""
        ms = []
        for name, m in Measurement.instances.items():
            if self.name in m.all_parameters:
                ms.append(name)
            else:
                for obs in m.all_parameters:
                    if isinstance(obs, tuple):
                        if obs[0] == self.name:
                            ms.append(name)
                            break
        return ms

    def theory_citations(self, *args, **kwargs):
        """Return a set of theory papers (in the form of INSPIRE texkeys) to
        cite for the theory prediction for an observable.

        Arguments are passed to the observable and are necessary,
        depending on the observable (e.g. $q^2$-dependent observables).
        """
        with flavio.citations.collect() as citations:
            flavio.sm_prediction(self.name, *args, **kwargs)
        return citations.set



class AuxiliaryQuantity(NamedInstanceClass):
    """An auxiliary quantity is something that can be computed theoretically but
    not measured directly, e.g. some sub-contribution to an amplitude or a form
    factor."""

    def __init__(self, name, arguments=None):
        super().__init__(name)
        self.arguments = arguments

    def get_implementation(self):
        try:
            implementation_name = config['implementation'][self.name]
        except KeyError:
            raise KeyError("No implementation specified for auxiliary quantity " + self.name)
        return Implementation[implementation_name]

    def prediction_central(self, constraints_obj, wc_obj, *args, **kwargs):
        implementation = self.get_implementation()
        return implementation.get_central(constraints_obj, wc_obj, *args, **kwargs)

    def prediction(self, par_dict, wc_obj, *args, **kwargs):
        implementation = self.get_implementation()
        return implementation.get(par_dict, wc_obj, *args, **kwargs)


class Prediction(object):
    """A prediction is the theoretical prediction for an observable."""

    def __init__(self, observable, function):
        try:
            Observable[observable]
        except KeyError:
            raise ValueError("The observable " + observable + " does not exist")
        self.observable = observable
        self.function = function
        self.observable_obj = Observable[observable]
        self.observable_obj.set_prediction(self)

    def get_central(self, constraints_obj, wc_obj, *args, **kwargs):
        par_dict = constraints_obj.get_central_all()
        fwc_obj = flavio.WilsonCoefficients.from_wilson(wc_obj, par_dict)
        return self.function(fwc_obj, par_dict, *args, **kwargs)

    def get_par(self, par_dict, wc_obj, *args, **kwargs):
        fwc_obj = flavio.WilsonCoefficients.from_wilson(wc_obj, par_dict)
        return self.function(fwc_obj, par_dict, *args, **kwargs)


class Implementation(NamedInstanceClass):
    """An implementation is the theoretical prediction for an auxiliary
    quantity."""

    @classmethod
    def show_all(cls):
        all_dict = {}
        for name in cls.instances:
            inst = cls[name]
            quant = inst.quantity
            descr = inst.description
            all_dict[quant] = {name: descr}
        return all_dict

    def __init__(self, name, quantity, function):
        super().__init__(name)
        try:
            AuxiliaryQuantity[quantity]
        except KeyError:
            raise ValueError("The quantity " + quantity + " does not exist")
        self.quantity = quantity
        self.function = function
        self.quantity_obj = AuxiliaryQuantity[quantity]

    def get_central(self, constraints_obj, wc_obj, *args, **kwargs):
        par_dict = constraints_obj.get_central_all()
        fwc_obj = flavio.WilsonCoefficients.from_wilson(wc_obj, par_dict)
        return self.function(fwc_obj, par_dict, *args, **kwargs)

    def get_random(self, constraints_obj, wc_obj, *args, **kwargs):
        par_dict = constraints_obj.get_random_all()
        fwc_obj = flavio.WilsonCoefficients.from_wilson(wc_obj, par_dict)
        return self.function(fwc_obj, par_dict, *args, **kwargs)

    def get(self, par_dict, wc_obj, *args, **kwargs):
        fwc_obj = flavio.WilsonCoefficients.from_wilson(wc_obj, par_dict)
        return self.function(fwc_obj, par_dict, *args, **kwargs)


class Measurement(Constraints, NamedInstanceClass):
    """A (experimental) measurement associates one (or several) probability
    distributions to one (or several) observables. If it contains several
    observables, these can (but do not have to) be correlated.

    To instantiate the class, call Measurement(name) with a string uniquely
    describing the measurement (e.g. 'CMS Bs->mumu 2012').

    To add a constraint (= central vaue(s) and uncertainty(s)), use

    `add_constraint(observables, constraint)`

    where `constraint` is an instance of a descendant of
    ProbabilityDistribution and `observables` is a list of either
     - a string observable name in the case of observables without arguments
     - or a tuple `(name, x_1, ..., x_n)`, where the `x_i` are float values for
       the arguments, of an observable with `n` arguments.
    """

    def __init__(self, name):
        NamedInstanceClass.__init__(self, name)
        Constraints.__init__(self)
        self.inspire = ''
        self.experiment = ''
        self.url = ''

    def __repr__(self):
        return "Measurement('{}')".format(self.name)

    def _repr_markdown_(self):
        md = "### Measurement `{}`\n\n".format(self.name)
        if self.experiment:
            md += "Experiment: {}\n\n".format(self.experiment)
        if self.inspire:
            md += ("[Inspire](http://inspirehep.net/search?&p=texkey+{})\n\n"
                   .format(urllib.parse.quote(self.inspire)))
        if self.url:
            md += "URL: <{}>\n\n".format(self.url)
        if self.description:
            md += "Description: {}\n\n".format(self.description)
        if self.all_parameters:
            md += "Measured observables:\n\n"
            for obs in self.all_parameters:
                if isinstance(obs, tuple):
                    name = obs[0]
                    args = obs[1:]
                    argnames = Observable[name].arguments
                    md += "- {}".format(Observable[name].tex)
                    for i, arg in enumerate(args):
                        md += ", `{}` = {}".format(argnames[i], arg)
                    md += "\n"
                else:
                    md += "- {}\n".format(Observable[obs].tex)
        return md
