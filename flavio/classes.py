"""Base classes for `flavio`"""


import numpy as np
from .config import config
from collections import OrderedDict
import copy
import math
from flavio._parse_errors import constraints_from_string, convolve_distributions, errors_from_string, string_from_constraints
import scipy.stats

def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class NamedInstanceClass(object):
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
    to objects like parameters or measurement. This is the base class of
    ParameterConstraints (that holds the numerical values and uncertainties
    of all the parameters) and Measurements (that holds the numerical values
    and uncertainties of all the experimental measurements.)

    Since this class is not meant for direct use, see these child classes for
    documentation.
    """

    def __init__(self):
            # Here we have two ordered dictionaries. _constraints has the form
            # { <constraint1>: [parameter1, parameter2, ...], <constraint2>: ...}
            # where the <constraint>s are instances of ProbabilityDistribution
            # and the parameters string names, while _parameters has the form
            # { parameter1: [(num1, <constraint1>)]} where num1 is 0 for a
            # univariate constraints and otherwise gives the position of
            # parameter1 in the multivariate vector.
            # In summary, having these to dicts allow a bijective mapping between
            # constraints (that might apply to multiple parameters) and parameters.
        self._constraints = OrderedDict()
        self._parameters = OrderedDict()

    @property
    def all_parameters(self):
        """Returns a list of all parameters/observables constrained."""
        return list(self._parameters.keys())

    def add_constraint(self, parameters, constraint):
        """Add a constraint to the parameter/observable.

        `constraint` must be an instance of a child of ProbabilityDistribution.

        Note that if there already exists a constraint, it will be removed."""
        for num, parameter in enumerate(parameters):
            # remove constraints if there are any
            if parameter in self._parameters:
                self.remove_constraints(parameter)
        # populate the dictionaries defined in __init__
            self._parameters[parameter] = [(num, constraint)]
        self._constraints[constraint] = parameters

    def set_constraint(self, parameter, constraint_string):
        """Set the constraints on a parameter/observable by specifying a string
        that can be e.g. of the form 1.55(3)(1) or 4.0±0.1. Existing
        constraints will be removed."""
        pds = constraints_from_string(constraint_string)
        combined_pd = convolve_distributions(pds)
        self.add_constraint([parameter], combined_pd)

    def get_constraint_string(self, parameter):
        """Return a string of the form 4.0±0.1±0.3 for the constraints on
        the parameter. Correlations are ignored."""
        pds = self._parameters[parameter]
        return string_from_constraints(pds)

    def remove_constraints(self, parameter):
        """Remove all constraints on a parameter."""
        self._parameters[parameter] = []

    def get_central(self, parameter):
        """Get the central value of a parameter"""
        if parameter not in self._parameters.keys():
            raise ValueError('No constraints applied to parameter/observable ' + self.parameter)
        else:
            num, constraint = self._parameters[parameter][0]
            # return the num-th entry of the central value vector
            return np.ravel([constraint.central_value])[num]

    def get_central_all(self):
        """Get central values of all constrained parameters."""
        return {parameter: self.get_central(parameter) for parameter in self._parameters.keys()}

    def get_random_all(self):
        """Get random values for all constrained parameters where they are
        distributed according to the probability distributions applied."""
        # first, generate random values for every single one of the constraints
        random_constraints = [constraint.get_random() for constraint in self._constraints.keys()]
        random_dict = {}
        # now, iterate over the parameters
        for parameter, constraints in self._parameters.items():
            # here, the idea is the following. Assume there is a parameter
            # p with central value c and N probability distributions that have
            # random values r_1, ..., r_N (obtained above). The final random
            # value for p is then given by p_r = c + \sum_i^N (r_i - c).
            central_value =  self.get_central(parameter)
            random_dict[parameter] = central_value  # step 1: p_r = c
            for num, constraint in constraints:
                idx = list(self._constraints.keys()).index(constraint)
                # step 1+i: p_r += r_i - c
                random_dict[parameter] += np.ravel([random_constraints[idx]])[num] - central_value
        return random_dict

    def get_1d_errors(self, N=1000):
        """Get the Gaussian standard deviation for every parameter/observable
        obtained by generating N random values.."""
        random_dict_list = [self.get_random_all() for i in range(N)]
        interval_dict = {}
        for k in random_dict_list[0].keys():
            arr = np.array([r[k] for r in random_dict_list])
            interval_dict[k] = np.std(arr)
        return interval_dict

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
          that should be ignored. In practice, this is done by setting these
          parameters equal to their central values. This way, they contribute
          a constant shift to the log probability, but their values in
          par_dict play no role.
        """
        prob_dict = {}
        for constraint, parameters in self._constraints.items():
            def constraint_central_value(constraint, parameters, parameter):
                # this function is required to get the central value for the
                # excluded_parameters, consistently for univariate and multivariate
                # distributions.
                if len(parameters) == 1:
                    # for univariate, it's trivial
                    return constraint.central_value
                else:
                    # for multivariate, need to find the position of the parameter
                    # in the vector and return the appropriate entry
                    return constraint.central_value[parameters.index(parameter)]
            # construct the vector of values from the par_dict, replaced by central values in the case of excluded_parameters
            x = [par_dict[p]
                if (p not in exclude_parameters
                    and (parameters.index(p), constraint) in self._parameters[p])
                else constraint_central_value(constraint, parameters, p)
                for p in parameters]
            if len(x) == 1:
                # 1D constraints should have a scalar, not a length-1 array
                x = x[0]
            prob_dict[constraint] = constraint.logpdf(x)
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


########## Observable Class ##########
class Observable(NamedInstanceClass):
    """An Observable is something that can be measured experimentally and
    predicted theoretically."""

    def __init__(self, name, arguments=None):
        super().__init__(name)
        self.arguments = arguments
        self.prediction = None
        self.tex = ''

    def set_prediction(self, prediction):
        self.prediction = prediction

    def prediction_central(self, constraints_obj, wc_obj, *args, **kwargs):
        return self.prediction.get_central(constraints_obj, wc_obj, *args, **kwargs)

    def prediction_par(self, par_dict, wc_obj, *args, **kwargs):
        return self.prediction.get_par(par_dict, wc_obj, *args, **kwargs)


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
        return Implementation.get_instance(implementation_name)

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
            Observable.get_instance(observable)
        except KeyError:
            raise ValueError("The observable " + observable + " does not exist")
        self.observable = observable
        self.function = function
        self.observable_obj = Observable.get_instance(observable)
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
            inst = cls.get_instance(name)
            quant = inst.quantity
            descr = inst.description
            all_dict[quant] = {name: descr}
        return all_dict

    def __init__(self, name, quantity, function):
        super().__init__(name)
        try:
            AuxiliaryQuantity.get_instance(quantity)
        except KeyError:
            raise ValueError("The quantity " + quantity + " does not exist")
        self.quantity = quantity
        self.function = function
        self.quantity_obj = AuxiliaryQuantity.get_instance(quantity)

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
