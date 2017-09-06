import scipy
import numpy as np
from scipy.optimize import minimize

def minimize_robust(fun, x0, args=(), methods=None, tries=3, disp=False, **kwargs):
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
        methods = ('SLSQP', 'BFGS', 'Nelder-Mead')
    for i in range(tries):
        for m in methods:
            if disp:
                print("Starting try no. {} with method {}".format(i+1, m))
                print("Current function value: {}".format(_val))
            if 'options' in kwargs:
                kwargs['options'].update({'disp': disp})
            else:
                kwargs['options'] = {'disp': disp}
            opt = minimize(fun, x0, args=args, method=m, **kwargs)
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
