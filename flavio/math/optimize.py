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
                opt = minimize_migrad(fun, x0, args=args, print_level=int(disp))
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


class MinuitFunction(object):
    """Function wrapper for Minuit to allow supplying function with additional
    arguments"""
    def __init__(self, f, args=()):
        """Initialize the instance. f: function"""
        import iminuit
        self.f = f
        self.args = args
        self.func_code = iminuit.util.make_func_code('x')

    def __call__(self, x):
        return self.f(x, *self.args)

def minimize_migrad(fun, x0, args=(), print_level=0):
    """Minimization function using MINUIT's MIGRAD minimizer."""
    import iminuit
    mfun = MinuitFunction(f=fun, args=args)
    # run
    minuit = iminuit.Minuit(mfun, x0)
    minuit.errordef = iminuit.Minuit.LEAST_SQUARES # == 1
    minuit.print_level = print_level
    mres = minuit.migrad()
    # cast migrad result in terms of scipy-like result object
    res = scipy.optimize.OptimizeResult()
    res.success = mres.fmin.is_valid
    res.fun = mres.fmin.fval
    res.x = np.array(mres.values)
    res.nfev = mres.fmin.nfcn
    return res
