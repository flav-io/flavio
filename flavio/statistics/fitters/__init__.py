"""Fitters are classes that take an instance of a fit and determine the
preferred regions for the parameters of interest, e.g. via Bayesian
marginalization.

Note that this module has been deprecated as of flavio v1.6.0
and will not be developed further as part of flavio.
"""

import warnings
warnings.warn("The `flavio.statistics.fitters` module has been deprecated"
              " as of flavio v1.6.0.",
              DeprecationWarning, stacklevel=2)
