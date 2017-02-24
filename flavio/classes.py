"""Base classes for `flavio`"""


import numpy as np
from .config import config
from collections import OrderedDict, defaultdict
import copy
import math
from flavio._parse_errors import constraints_from_string, convolve_distributions, errors_from_string
from flavio.statistics.probability import class_from_string
import scipy.stats
import warnings

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

    def set_description(self, description):
        self.description = description



########## Parameter Class ##########
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


########## Constraints Class ##########
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
        unceretainty) are given, the total constraint will be the convolution
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
            if isinstance(constraint_dict, dict):
                dict_list = [constraint_dict]
            else:
                dict_list = constraint_dict
            pds = []
            for d in dict_list:
                dist = class_from_string[d['distribution']]
                pds.append(dist(**{k: float(v) for k, v in d.items() if k!='distribution'}))
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
            # return the num-th entry of the central value vector
            return np.ravel([constraint.central_value])[num]

    def get_central_all(self):
        """Get central values of all constrained parameters."""
        return {parameter: self.get_central(parameter) for parameter in self._parameters.keys()}

    def get_random_all(self):
        """Get random values for all constrained parameters where they are
        distributed according to the probability distribution applied."""
        # first, generate random values for every single one of the constraints
        random_constraints = [constraint.get_random() for constraint, _ in self._constraints]
        random_dict = {}
        # now, iterate over the parameters
        for parameter, constraints in self._parameters.items():
            num, constraint = constraints
            idx = ([constraint for constraint, _ in self._constraints]).index(constraint)
            random_dict[parameter] = np.ravel([random_constraints[idx]])[num]
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
            error_dict[parameter] = (
                                        np.ravel([errors_right[idx]])[num],
                                        np.ravel([errors_left[idx]])[num]
                                    )
        return error_dict

    def get_logprobability_all(self, par_dict, exclude_parameters=[]):
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
            x = [par_dict[p]
                for p in parameters
                if (p not in exclude_parameters
                    and (parameters.index(p), constraint) == self._parameters.get(p, None))]
            if not x:
                # nothing to constrain
                continue
            if len(parameters) == 1:
                # 1D constraints should have a scalar, not a length-1 array
                prob_dict[constraint] = constraint.logpdf(x[0])
            else:
                # for multivariate distributions
                if len(x) == len(parameters):
                    # no parameter has been excluded
                    exclude = None
                else:
                    exclude = tuple(i
                        for i, p in enumerate(parameters)
                        if (p in exclude_parameters
                            or (parameters.index(p), constraint) != self._parameters.get(p, None)))
                prob_dict[constraint] = constraint.logpdf(x, exclude=exclude)
        return prob_dict

    def copy(self):
        # this is to have a .copy() method like for a dictionary
        return copy.deepcopy(self)


########## ParameterConstraints Class ##########
class ParameterConstraints(Constraints):
    """
    """

    def __init__(self):
        super().__init__()

########## WilsonCoefficientPriors Class ##########
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

########## Observable Class ##########
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
        taxonomy_list = taxonomy_string.split(' :: ') + [ self.name ]
        t = self.__class__.taxonomy
        for node in taxonomy_list:
            t = t[node]

    @classmethod
    def taxonomy_dict(cls):
        """Return the hierarchical metadata taxonomy as a nested dictionary."""
        return dicts(cls.taxonomy)


########## AuxiliaryQuantity Class ##########
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



########## Prediction Class ##########
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
        return self.function(wc_obj, par_dict, *args, **kwargs)

    def get_par(self, par_dict, wc_obj, *args, **kwargs):
        return self.function(wc_obj, par_dict, *args, **kwargs)


########## Implementation Class ##########
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
        return self.function(wc_obj, par_dict, *args, **kwargs)

    def get_random(self, constraints_obj, wc_obj, *args, **kwargs):
        par_dict = constraints_obj.get_random_all()
        return self.function(wc_obj, par_dict, *args, **kwargs)

    def get(self, par_dict, wc_obj, *args, **kwargs):
        return self.function(wc_obj, par_dict, *args, **kwargs)


########## Measurement Class ##########
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
     - or a tuple `(name, x_1, ..., x_n)`, where the `x_i` are float values for the
       arguments, of an observable with `n` arguments.
    """

    def __init__(self, name):
        NamedInstanceClass.__init__(self, name)
        Constraints.__init__(self)
        self.inspire = ''
        self.experiment = ''
        self.url = ''
