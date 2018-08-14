import scipy
import numpy as np
from scipy.optimize import minimize


def minimize_robust(fun, x0, args=(), methods=None, tries=3, disp=False,
                    **kwargs):
    """Minimization of scalar function of one or more variables.

    This is a wrapper around the `scipy.optimize.minimize` function that
    iterates in case optimization does not converge. Endpoints of unsuccessful
    tries are used as starting points of new tries. Returns a scipy
    `OptimizeResult`.

    New arguments compared to the scipy function:

    - methods: tuple of methods to try consecutively. By default, uses
      `('SLSQP', 'BFGS', 'Nelder-Mead')`
    - tries: number of tries before giving up. Defaults to 3.
    - disp: if True (default: False), print convergence information
    """
    _val = fun(x0, *args) # initial value of target function
    _x0 = x0[:] # copy initial x
    if methods is None:
        methods = ('SLSQP', 'MIGRAD', 'BFGS', 'Nelder-Mead')
    for i in range(tries):
        for m in methods:
            if disp:
                print("Starting try no. {} with method {}".format(i+1, m))
                print("Current function value: {}".format(_val))
            if m == 'MIGRAD':
                try:
                    import iminuit
                except ImportError:
                    if disp:
                        print("Skipping method MIGRAD: no iminuit installation found.")
                    continue
                options = {'print_level': int(disp)} # 0 or 1
                options.update(kwargs)
                opt = minimize_migrad(fun, x0, args=args, **options)
            else:
                options = {'disp': disp}
                options.update(kwargs)
                opt = minimize(fun, x0, args=args, method=m, options=options)
            if opt.success:
                return opt
            elif opt.fun < _val:
                if disp:
                    print("Optimization did not converge.")
                    print("Current function value: {}".format(_val))
                # if this step actually led to some improvement,
                # use endpoint as new starting point
                _x0 = opt.x
    return opt

def maximize_robust(fun, x0, args=(), methods=None, tries=3, disp=False, **kwargs):
    """Maximization of scalar function of one or more variables.

    See `minimize_robust` for details.
    """
    def mfun(*args):
        return -fun(*args)
    res = minimize_robust(mfun, x0,
                           args=args, methods=methods,
                           tries=tries, disp=disp, **kwargs)
    res.fun = -res.fun # change back sign for function value
    return res


class AttributeDict(dict):
    """Dictionary subclass with attribute access"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class MinuitFunction(object):
    """Function wrapper for Minuit to allow supplying function with vector
    arguments"""
    def __init__(self, f, dim, args=()):
        """Initialize the instance. f: function, dim: number of dimensions"""
        self.f = f
        self.dim = dim
        self.args = args

    @property
    def __code__(self):
        """Needed to fake the function signature for Minuit"""
        d = AttributeDict()
        d.co_varnames = ['x{}'.format(i) for i in range(self.dim)]
        d.co_argcount = len(d.co_varnames)
        return d

    def __call__(self, *x):
        return self.f(x, *self.args)


def minimize_migrad(fun, x0, args=(), dx0=None, **kwargs):
    """Minimization function using MINUIT's MIGRAD minimizer."""
    import iminuit
    mfun = MinuitFunction(f=fun, dim=len(x0), args=args)
    # bring the parameters in a suitable form
    par = iminuit.util.describe(mfun)
    x0_dict = {par[i]: x0i for i, x0i in enumerate(x0)}
    if dx0 is None:
        dx0 = np.ones(len(x0))
    dx0_dict = {'error_' + par[i]: dx0i for i, dx0i in enumerate(dx0)}
    # run
    minuit_args={'errordef': 1}
    minuit_args.update(kwargs)
    minuit = iminuit.Minuit(mfun, **x0_dict, **dx0_dict, **minuit_args)
    fmin, param = minuit.migrad()
    # cast migrad result in terms of scipy-like result object
    res = scipy.optimize.OptimizeResult()
    res.success = fmin['is_valid']
    res.fun = fmin['fval']
    res.x = np.array([p['value'] for p in param])
    res.nfev = fmin['nfcn']
    return res
