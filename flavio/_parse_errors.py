import re
from flavio.statistics.probability import *

# for strings of the form '< 5.3e-8 @ 95% CL'
_pattern_upperlimit = re.compile(r"^\s*<\s*([-+]?\d+\.?\d*)([eE][-+]?\d+)?\s*@\s*(\d+\.?\d*)\s*\%\s*C[\.\s]*L[\.\s]*$")
# for strings of the form '1.67(3)(5) 1e-3'
_pattern_brackets = re.compile(r"^\(?\s*(-?\d+\.?\d*)\s*((?:\(\s*\d+\.?\d*\s*\)\s*)+)\)?\s*\*?\s*(?:(?:e|E|1e|1E|10\^)\(?([+-]?\d+)\)?)?$")
# for strings of the form '(1.67 +- 0.3 +- 0.5) * 1e-3'
_pattern_plusminus = re.compile(r"^\(?\s*(-?\d+\.?\d*)\s*((?:[+\-±\\pm]+\s*\d+\.?\d*\s*)+)\)?\s*\*?\s*(?:(?:e|E|1e|1E|10\^)\(?([+-]?\d+)\)?)?$")


def errors_from_string(constraint_string):
    """Convert a string like '1.67(3)(5)' or '1.67+-0.03+-0.05' to a dictionary
    of central values errors."""
    m = _pattern_brackets.match(constraint_string)
    if m is None:
        m = _pattern_plusminus.match(constraint_string)
    if m is None:
        raise ValueError("Constraint " + constraint_string + " not understood")
    # extracting the central value and overall power of 10
    if m.group(3) is None:
        overall_factor = 1
    else:
        overall_factor = 10**float(m.group(3))
    central_value = m.group(1)
    # number_decimal gives the number of digits after the decimal point
    if len(central_value.split('.')) == 1:
        number_decimal = 0
    else:
        number_decimal = len(central_value.split('.')[1])
    central_value = float(central_value) * overall_factor
    # now, splitting the errors
    error_string = m.group(2)
    pattern_brackets_err = re.compile(r"\(\s*(\d+\.?\d*)\s*\)\s*")
    pattern_symmetric_err = re.compile(r"(?:±|\\pm|\+\-)(\s*\d+\.?\d*)")
    pattern_asymmetric_err = re.compile(r"\+\s*(\d+\.?\d*)\s*\-\s*(\d+\.?\d*)")
    errors = {}
    errors['central_value'] = central_value
    errors['symmetric_errors'] = []
    errors['asymmetric_errors'] = []
    if pattern_brackets_err.match(error_string):
        for err in re.findall(pattern_brackets_err, error_string):
            if not err.isdigit():
                # if isdigit() is false, it means that it is a number
                # with a decimal point (e.g. '1.5'), so no rescaling is necessary
                standard_deviation = float(err)*overall_factor
            else:
                # if the error is just digits, need to rescale it by the
                # appropriate power of 10
                standard_deviation = float(err)*10**(-number_decimal)*overall_factor
            errors['symmetric_errors'].append(standard_deviation)
    elif pattern_symmetric_err.match(error_string) or pattern_asymmetric_err.match(error_string):
        for err in re.findall(pattern_symmetric_err, error_string):
            errors['symmetric_errors'].append( float(err)*overall_factor )
        for err in re.findall(pattern_asymmetric_err, error_string):
            right_err = float(err[0])*overall_factor
            left_err = float(err[1])*overall_factor
            errors['asymmetric_errors'].append((right_err, left_err))
    return errors

def limit_from_string(constraint_string):
    m = _pattern_upperlimit.match(constraint_string)
    if m is None:
        raise ValueError("Constraint " + constraint_string + " not understood")
    sg, ex, cl_pc = m.groups()
    if ex is None:
        limit = float(sg)
    else:
        limit = float(sg + ex)
    cl = float(cl_pc)/100.
    return limit, cl

def errors_from_constraints(probability_distributions):
  """Return a string of the form 4.0±0.1±0.3 for the constraints on
  the parameter. Correlations are ignored."""
  errors = {}
  errors['symmetric_errors'] = []
  errors['asymmetric_errors'] = []
  for num, pd in probability_distributions:
      errors['central_value'] = pd.central_value
      if isinstance(pd, DeltaDistribution):
          # delta distributions (= no error) can be skipped
          continue
      elif isinstance(pd, NormalDistribution):
          errors['symmetric_errors'].append(pd.standard_deviation)
      elif isinstance(pd, AsymmetricNormalDistribution):
          errors['asymmetric_errors'].append((pd.right_deviation, pd.left_deviation))
      elif isinstance(pd, MultivariateNormalDistribution):
          errors['central_value'] = pd.central_value[num]
          errors['symmetric_errors'].append(math.sqrt(pd.covariance[num, num]))
  return errors

def string_from_constraints(probability_distributions):
    errors = errors_from_constraints(probability_distributions)
    string = str(errors['central_value'])
    for err in errors['symmetric_errors']:
        string += ' ± ' + str(err)
    for right_err, left_err in errors['asymmetric_errors']:
        string += r' ^{+' + str(right_err) + r'}_{-' + str(left_err) + r'}'
    return string

def constraints_from_string(constraint_string):
    """Convert a string like '1.67(3)(5)' or '1.67+-0.03+-0.05' to a list
    of ProbabilityDistribution instances."""
    # first of all, replace dashes (that can come from copy-and-pasting latex) by minuses
    try:
        float(constraint_string)
        # if the string represents just a number, return a DeltaDistribution
        return [DeltaDistribution(float(constraint_string))]
    except ValueError:
        # first of all, replace dashes (that can come from copy-and-pasting latex) by minuses
        constraint_string = constraint_string.replace('−','-')
        # try again if the number is a float now
        try:
            float(constraint_string)
            return {'central_value': float(constraint_string)}
        except:
            pass
    if _pattern_upperlimit.match(constraint_string):
        limit, cl = limit_from_string(constraint_string)
        return [GaussianUpperLimit(limit, cl)]
    elif _pattern_brackets.match(constraint_string) or _pattern_plusminus.match(constraint_string):
        errors = errors_from_string(constraint_string)
        if 'symmetric_errors' not in errors and 'asymmetric_errors' not in errors:
            return [DeltaDistribution(errors['central_value'])]
        pd = []
        for err in errors['symmetric_errors']:
            pd.append(NormalDistribution(errors['central_value'], err))
        for err_right, err_left in errors['asymmetric_errors']:
            pd.append(AsymmetricNormalDistribution(errors['central_value'], err_right, err_left))
        return pd
    else:
        raise ValueError("Constraint " + constraint_string + " not understood")
