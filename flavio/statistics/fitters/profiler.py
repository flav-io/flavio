"""1D and 2D likelihood profilers for frequentist fits."""

import numpy as np
import flavio
import scipy.optimize
from flavio.math.optimize import minimize_robust, maximize_robust
import warnings

def par_shift_scale(par_obj, parameters):
    """Determine a shift and scale factor that rescales
    parameters to be centered at 0 and varying by O(1)."""
    # central nuisance parameters
    cen_all = par_obj.get_central_all()
    # shift everything to 0
    shift = np.array([-cen_all[p] for p in parameters])
    # errors of nuisance parameters
    err_all = par_obj.get_1d_errors_rightleft()
    err = [err_all[p] for p in parameters]
    def scalefac(er, el):
        if er==0 and el==0:
            # for nuisances with vanishing uncertainties: don't rescale
            return 1
        # example: if errors are -0.01, +0.01, rescale by 10=1/0.1
        return 1/max(abs(er), abs(er))
    scale = np.array([scalefac(er, el) for er, el in err])
    return shift, scale

def nuisance_shift_scale(fit):
    """Determine a shift and scale factor that rescales nuisance
    parameters to be centered at 0 and varying by O(1)."""
    return par_shift_scale(fit.par_obj, fit.nuisance_parameters)

def reshuffle_1d(x, i0):
    """Reshuffle 1D array to make index i0 the first index and append the
    lower indices to the end."""
    return np.hstack((x[i0:], x[:i0]))

def unreshuffle_1d(x, i0):
    """Undo the reshuffle_1d operation."""
    N = len(x)
    return np.hstack((np.asarray(x)[N-i0:].T, np.asarray(x)[:N-i0].T)).T

def reshuffle_2d(x, ij0):
    """Reshuffle 2D array to make index (ij0) the first index, flatten the
    array, and append the following indices in the end."""
    m, n = np.asarray(x).shape
    i0, j0 = ij0
    x_flat = np.asarray(x).ravel()
    i0_1d = n*i0 + j0
    return reshuffle_1d(x_flat, i0_1d), i0_1d

def unreshuffle_2d(x, i0, shape):
    """Undo the reshuffle_2d operation."""
    x_flat = unreshuffle_1d(x, i0)
    return np.reshape(x_flat, shape)


class Profiler(object):
    """Parent class for profilers. Not meant to be used directly."""
    def __init__(self, fit):
        """Initialize the profiler instance."""
        self.fit = fit
        assert isinstance(fit, flavio.statistics.fits.FrequentistFit), \
                    "Fit object must be an instance of FrequentistFit"
        self.nuisance_shift, self.nuisance_scale = nuisance_shift_scale(fit)
        self.n_fit_p = len(self.fit.fit_parameters)
        self.n_nui_p = len(self.fit.nuisance_parameters)
        self.n_wc = len(self.fit.fit_wc_names)
        self.bf = None
        self.log_profile_likelihood = None
        self.profile_nuisance = None

    def f_target(self, x_n, par_wc_fixed):
        """Target function (log likelihood) in terms of rescaled and shifted
        nuisance parameters x_n for given values of the fixed parameters/Wilson
        coefficients."""
        x = np.zeros(self.fit.dimension)
        if self.n_fit_p > 0:
            x[:self.n_fit_p] = par_wc_fixed[:self.n_fit_p]
        if self.n_wc > 0:
            x[-self.n_wc:] = par_wc_fixed[-self.n_wc:]
        x[self.n_fit_p:self.n_fit_p+self.n_nui_p] = x_n/self.nuisance_scale - self.nuisance_shift
        return self.fit.log_likelihood(x)

    def f_target_global(self, x):
        """Target function (log likelihood) in terms of fit parameters,
        rescaled and shifted nuisance parameters, and (unscaled) Wilson
        coefficients for global optimization."""
        X = np.array(x)
        if self.n_nui_p > 0:
            X[self.n_fit_p:self.n_fit_p+self.n_nui_p] = x[self.n_fit_p:self.n_fit_p+self.n_nui_p]/self.nuisance_scale - self.nuisance_shift
        return self.fit.log_likelihood(X)

    def best_fit(self, fitpar0, **kwargs):
        """Determine the global best-fit point in the space of fit parameters,
        nuisance parameters, and Wilson coefficients.

        Returns a scipy.optimize.OptimizeResult instance."""
        x0 = np.zeros(self.fit.dimension)
        if self.n_fit_p > 0:
            x0[:self.n_fit_p] = fitpar0
        res = maximize_robust(self.f_target_global, x0=x0, **kwargs)
        return res


class Profiler1D(Profiler):
    """1-dimensional likelihood profiler.


    Methods:

    - run: profile the likelihood in the 1D interval.
    - pvalue_prob: return the p-value under the assumption of Wilks' theorem
    - pvalue_prob_plotdata: return a dictionary suited to be fed into the
      `flavio.plots.pvalue_plot` plotting function
    """
    def __init__(self, fit, x_min, x_max):
        """Initialize the profiler instance.

        Arguments:

        - x_min: lower bound of the parameter/coefficient of interest
        - x_max: upper bound of the parameter/coefficient of interest
        """
        assert len(fit.fit_parameters) + len(fit.fit_wc_names) == 1, (
                    "Fit instance must have precisely one fit parameter"
                    " or one fitted Wilson coefficient")
        super().__init__(fit)
        assert x_min < x_max, "x_max must be bigger than x_min"
        self.x_min = x_min
        self.x_max = x_max
        self.x_bf = None
        self.x = None

    def run(self, steps=20, **kwargs):
        """Maximize the likelihood by varying the nuisance parameters.

        Arguments:

        - steps (defaults to 20): number of steps in the 1D interval of interest

        Additional keyword arguments will be passed to
        `flavio.math.optimize.maximize_robust`.

        Returns:

        `x, z, n` where
        - x: the points at which the likelihood has been maximized
        - z: the log likelihood at these points
        - n: the values of the nuisance parameters at these points; has shape
            (n_nuisance, steps)
        """
        # determine x-value at the global best-fit point
        try:
            bf = self.best_fit(fitpar0=self.fit.get_central_fit_parameters)
        except KeyError:
            # if the fit parameters do not have existing constraints, use center
            bf = self.best_fit(fitpar0=(self.x_max-self.x_min)/2)
        self.bf = bf
        if self.n_fit_p == 1:
            self.x_bf = bf.x[0] # best-fit parameter
        elif self.n_wc == 1:
            self.x_bf = bf.x[-1] # ... or best-fit Wilson coefficient
        if self.x_bf == self.x_min or self.x_bf == self.x_max:
            # if the best-fit value is at the border,
            # just make a linspace
            x = np.linspace(self.x_min, self.x_max, steps)
        else:
            # otherwise, divide the range into x<x_bf and x>x_bf,
            # with the same number of steps on both sides
            steps_l = steps//2
            steps_r = steps - steps_l + 1
            # [1:] is in order not to double count x_bf
            x = np.hstack([np.linspace(self.x_min, self.x_bf, steps_l),
                           np.linspace(self.x_bf, self.x_max, steps_r)[1:]])
        # determine index in x-array where the x is closest to x_bf
        i0 = (np.abs(x-self.x_bf)).argmin()
        x = reshuffle_1d(x, i0)
        z = np.zeros(steps)
        n = np.zeros((steps, self.n_nui_p))
        # start with global best-fit values for nuisance parameters
        x0 = bf.x[self.n_fit_p:self.n_fit_p+self.n_nui_p]
        for i, X in enumerate(x):
            res = maximize_robust(self.f_target, args=([X],), x0=x0, **kwargs)
            if res.success:
                z[i] = -res.fun # -f returns + the log likelihood
                n[i] = res.x/self.nuisance_scale - self.nuisance_shift
                # reset nuisances start values for next iteration to optimized values
                x0 = res.x
            else:
                z[i] = np.nan
        x = unreshuffle_1d(x, i0)
        z = unreshuffle_1d(z, i0)
        n = unreshuffle_1d(n, i0).T
        self.x = x
        self.log_profile_likelihood = z
        self.profile_nuisance = n
        return x, z, n

    def pvalue_prob(self):
        """Return p-value obtained under the assumption of Wilks' theorem.

        Requires the profiler to be run first using the `run` method."""
        if self.log_profile_likelihood is None:
            warnings.warn("You must run the profiler first.")
            return None
        chi2_dist = scipy.stats.chi2(1)
        chi2 = -2*self.log_profile_likelihood
        chi2_bf = -2*self.bf.fun
        # normally, the minimal chi2 should of course be the chi2 at the best-fit
        # point. If for reasons of numerical inaccuracy the chi2 array contains
        # a smaller value, use that.
        chi2_min = min(np.min(chi2), chi2_bf)
        delta_chi2 = chi2 - chi2_min
        cl = chi2_dist.cdf(delta_chi2)
        return 1-cl

    def pvalue_prob_plotdata(self):
        """Return a dictionary that can be fed into `flavio.plots.pvalue_plot`."""
        return {
                'x': self.x,
                'y': self.pvalue_prob(),
               }


class Profiler2D(Profiler):
    """2-dimensional likelihood profiler.


    Methods:

    - run: profile the likelihood in the 2D plane.
    - contour_plotdata: return a dictionary suited to be fed into the
      `flavio.plots.contour` plotting function
    """
    def __init__(self, fit, x_min, x_max, y_min, y_max):
        """Initialize the profiler instance.

        Arguments:

        - x_min, x_max: lower and upper bounds of x
        - y_min, y_max: lower and upper bounds of y
        """
        assert len(fit.fit_parameters) + len(fit.fit_wc_names) == 2, (
                    "Fit instance must have precisely one fit parameter"
                    " or one fitted Wilson coefficient")
        super().__init__(fit)
        assert x_min < x_max, "x_max must be bigger than x_min"
        assert y_min < y_max, "y_max must be bigger than y_min"
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def run(self, steps=(10, 10), **kwargs):
        """Maximize the likelihood by varying the nuisance parameters.

        Arguments:

        - steps: number of steps in the in the x and y direction.
          Tuple of length 2 that defaults to (10, 10)
        - method: minimization method to be used by scipy.optimize.minimize

        Returns:

        `x, y, z, n` where
        - x, y: the points at which the likelihood has been maximized
        - z: the log likelihood at these points
        - n: the values of the nuisance parameters at these points; has shape
          (n_nuisance, steps_x, steps_y)
        """
        # determine x- and y-value at the global best-fit point
        try:
            bf = self.best_fit(fitpar0=self.fit.get_central_fit_parameters)
        except KeyError:
            bf = self.best_fit(fitpar0=[(self.x_max-self.x_min)/2,
                                        (self.y_max-self.y_min)/2])
        self.bf = bf
        if self.n_wc > 0:
            x_bf, y_bf = np.hstack((bf.x[:self.n_fit_p], bf.x[-self.n_wc:]))
        else:
            x_bf, y_bf = bf.x[:self.n_fit_p]
        x = np.linspace(self.x_min, self.x_max, steps[0])
        y = np.linspace(self.y_min, self.y_max, steps[1])
        # determine index in x and y-arrays where the x/y is closest to x_bf/y_bf
        ij0 = (np.abs(x-x_bf)).argmin(), (np.abs(x-x_bf)).argmin()
        z = np.zeros(steps)
        n = np.zeros((self.n_nui_p, steps[0], steps[1]))
        xx, yy = np.meshgrid(x, y, indexing='ij')
        xx, i0_1d = reshuffle_2d(xx, ij0)
        yy, i0_1d = reshuffle_2d(yy, ij0)
        z, i0_1d = reshuffle_2d(z, ij0)
        n = np.array([reshuffle_2d(ni, ij0)[0] for ni in n])
        # start with global best-fit values for nuisance parameters
        x0 = bf.x[self.n_fit_p:self.n_fit_p+self.n_nui_p]
        for i in range(len(z)):
            res = maximize_robust(self.f_target, args=([xx[i], yy[i]],), x0=x0, **kwargs)
            if res.success:
                z[i] = -res.fun # -f returns + the log likelihood
                n[:,i] = res.x/self.nuisance_scale - self.nuisance_shift
                # reset nuisances start values for next iteration to optimized values
                x0 = res.x
            else:
                z[i] = np.nan
        xx = unreshuffle_2d(xx, i0_1d, steps)
        yy = unreshuffle_2d(yy, i0_1d, steps)
        x = xx[:,0]
        y = yy[0]
        z = unreshuffle_2d(z, i0_1d, steps)
        n = np.array([unreshuffle_2d(ni, i0_1d, steps) for ni in n])
        self.x = x
        self.y = y
        self.log_profile_likelihood = z
        self.profile_nuisance = n
        return x, y, z, n

    def contour_plotdata(self, n_sigma=(1,2)):
        """Return a dictionary that can be fed into `flavio.plots.contour`.

        Parameters:

        - `n_sigma`: tuple with integer sigma values which should be plotted.
          Defaults to (1, 2).
        """
        deltachi2 = -2*(self.log_profile_likelihood
                        -np.max(self.log_profile_likelihood))
        x, y = np.meshgrid(self.x, self.y, indexing='ij')
        return {
                'x': x,
                'y': y,
                'z': deltachi2,
                'levels': tuple(flavio.statistics.functions.delta_chi2(n, 2)
                                for n in n_sigma),
               }
