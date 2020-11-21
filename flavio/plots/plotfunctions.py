from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import flavio
from flavio.statistics.functions import delta_chi2, confidence_level
import scipy.optimize
import scipy.interpolate
import scipy.stats
from numbers import Number
from math import sqrt
import warnings
import inspect
from multiprocessing import Pool
from pickle import PicklingError
from flavio.plots.colors import lighten_color, get_color


def error_budget_pie(err_dict, other_cutoff=0.03):
    """Pie chart of an observable's error budget.

    Parameters:

    - `err_dict`: Dictionary as return from `flavio.sm_error_budget`
    - `other_cutoff`: If an individual error contribution divided by the total
      error is smaller than this number, it is lumped under "other". Defaults
      to 0.03.

    Note that for uncorrelated parameters, the total uncertainty is the squared
    sum of the individual uncertainties, so the relative size of the wedges does
    not correspond to the relative contribution to the total uncertainty.

    If the uncertainties of individual parameters are correlated, the total
    uncertainty can be larger or smaller than the squared sum of the individual
    uncertainties, so the representation can be misleading.
    """
    err_tot = sum(err_dict.values()) # linear sum of individual errors
    err_dict_sorted = OrderedDict(sorted(err_dict.items(), key=lambda t: -t[1]))
    labels = []
    fracs = []
    small_frac = []
    for key, value in err_dict_sorted.items():
        frac = value/err_tot
        if frac > other_cutoff:
            if isinstance(key, str):
                try:
                    labels.append(flavio.Parameter[key].tex)
                except KeyError:
                    # if 'key' is not actually a parameter (e.g. manually set by the user)
                    labels.append(key)
            elif isinstance(key, tuple):
                key_strings = [flavio.Parameter[k].tex for k in key]
                labels.append(', '.join(key_strings))
            fracs.append(frac)
        else:
            small_frac.append(frac)
    if small_frac:
        labels.append('other')
        # the fraction for the "other" errors is obtained by adding them in quadrature
        fracs.append(np.sqrt(np.sum(np.array(small_frac)**2)))
    # initially, the fractions had been calculated assuming that they add to
    # one, but adding the "other" errors in quadrature changed that - correct
    # all the fractions to account for this
    corr = sum(fracs)
    fracs = [f/corr for f in fracs]
    def my_autopct(pct):
        return r'{p:.2g}\%'.format(p=pct*err_tot)
    plt.axis('equal')
    return plt.pie(fracs,
        labels=labels,
        autopct=my_autopct,
        wedgeprops={'linewidth':0.5},
        colors=[lighten_color('C{}'.format(i), 0.5) for i in range(10)]
    )


def diff_plot_th(obs_name, x_min, x_max, wc=None, steps=100, scale_factor=1, **kwargs):
    r"""Plot the central theory prediction of an observable dependending on
    a continuous parameter, e.g. $q^2$.

    Parameters:

    - `x_min`, `x_max`: minimum and maximum values of the parameter
    - `wc` (optional): `WilsonCoefficient` instance to define beyond-the-SM
      Wilson coefficients
    - `steps` (optional): number of steps in x. Defaults to 100. Less is
      faster but less precise.
    - `scale_factor` (optional): factor by which all values will be multiplied.
      Defaults to 1.

    Additional keyword arguments are passed to the matplotlib plot function,
    e.g. 'c' for colour.
    """
    obs = flavio.classes.Observable[obs_name]
    if not obs.arguments or len(obs.arguments) != 1:
        raise ValueError(r"Only observables that depend on a single parameter are allowed")
    x_arr = np.arange(x_min, x_max, (x_max-x_min)/(steps-1))
    if wc is None:
        wc = flavio.physics.eft._wc_sm # SM Wilson coefficients
        obs_arr = [flavio.sm_prediction(obs_name, x) for x in x_arr]
    else:
        obs_arr = [flavio.np_prediction(obs_name, wc, x) for x in x_arr]
    ax = plt.gca()
    if 'c' not in kwargs and 'color' not in kwargs:
        kwargs['c'] = 'k'
    ax.plot(x_arr, scale_factor * np.asarray(obs_arr), **kwargs)


def diff_plot_th_err(obs_name, x_min, x_max, wc=None, steps=100,
                        steps_err=5, N=100, threads=1, label=None,
                        plot_args=None, fill_args=None,
                        scale_factor=1):
    r"""Plot the theory prediction of an observable dependending on
    a continuous parameter, e.g. $q^2$,
    with uncertainties as a function of this parameter.

    Parameters:

    - `x_min`, `x_max`: minimum and maximum values of the parameter
    - `wc` (optional): `WilsonCoefficient` instance to define beyond-the-SM
      Wilson coefficients
    - `steps` (optional): number of steps for the computation of the
      central value. Defaults to 100. Less is faster but less precise.
    - `steps_err` (optional): number of steps for the computation of the
      uncertainty. Defaults to 5 and should be at least 3. Larger is slower
      but more precise. See caveat below.
    - `N` (optional): number of random evaluations to determine the uncertainty.
      Defaults to 100. Less is faster but less precise.
    - `threads` (optional): if bigger than 1, number of threads to use for
      parallel computation of uncertainties
    - `plot_args` (optional): dictionary with keyword arguments to be passed
      to the matplotlib plot function, e.g. 'c' for colour.
    - `fill_args` (optional): dictionary with keyword arguments to be passed
      to the matplotlib fill_between function, e.g. 'facecolor'
    - `scale_factor` (optional): factor by which all values will be multiplied.
      Defaults to 1.

    A word of caution regarding the `steps_err` option. By default, the
    uncertainty is only computed at 10 steps and is interpolated in
    between. This can be enough if the uncertainty does not vary strongly
    with the parameter. However, when the starting point or end point of the plot range
    is outside the physical phase space, the uncertainty will vanish at that
    point and the interpolation might be inaccurate.
    """
    obs = flavio.classes.Observable[obs_name]
    if not obs.arguments or len(obs.arguments) != 1:
        raise ValueError(r"Only observables that depend on a single parameter are allowed")
    step = (x_max-x_min)/(steps-1)
    x_arr = np.arange(x_min, x_max+step, step)
    step = (x_max-x_min)/(steps_err-1)
    x_err_arr = np.arange(x_min, x_max+step, step)
    # fix to avoid bounds_error in interp1d due to lack of numerical precision
    x_err_arr[-1] = x_arr[-1]
    if wc is None:
        wc = flavio.physics.eft._wc_sm # SM Wilson coefficients
        obs_err_arr = [flavio.sm_uncertainty(obs_name, x, threads=threads) for x in x_err_arr]
        obs_arr = [flavio.sm_prediction(obs_name, x) for x in x_arr]
    else:
        obs_err_arr = [flavio.np_uncertainty(obs_name, wc, x, threads=threads) for x in x_err_arr]
        obs_arr = [flavio.np_prediction(obs_name, wc, x) for x in x_arr]
    ax = plt.gca()
    plot_args = plot_args or {}
    fill_args = fill_args or {}
    if label is not None:
        plot_args['label'] = label
    if 'alpha' not in fill_args:
        fill_args['alpha'] = 0.5
    ax.plot(x_arr, scale_factor * np.asarray(obs_arr), **plot_args)
    interp_err = scipy.interpolate.interp1d(x_err_arr, obs_err_arr,
                                            kind='quadratic')
    obs_err_arr_int = interp_err(x_arr)
    ax.fill_between(x_arr,
                    scale_factor * np.asarray(obs_arr - obs_err_arr_int),
                    scale_factor * np.asarray(obs_arr + obs_err_arr_int),
                    **fill_args)


def bin_plot_th(obs_name, bin_list, wc=None, divide_binwidth=False, N=50, threads=1, **kwargs):
    r"""Plot the binned theory prediction with uncertainties of an observable
    dependending on a continuous parameter, e.g. $q^2$ (in the form of coloured
    boxes).

    Parameters:

    - `bin_list`: a list of tuples containing bin boundaries
    - `wc` (optional): `WilsonCoefficient` instance to define beyond-the-SM
      Wilson coefficients
    - `divide_binwidth` (optional): this should be set to True when comparing
      integrated branching ratios from experiments with different bin widths
      or to theory predictions for a differential branching ratio. It will
      divide all values and uncertainties by the bin width (i.e. dimensionless
      integrated BRs will be converted to $q^2$-integrated differential BRs with
      dimensions of GeV$^{-2}$). Defaults to False.
    - `N` (optional): number of random draws to determine the uncertainty.
      Defaults to 50. Larger is slower but more precise. The relative
      error of the theory uncertainty scales as $1/\sqrt{2N}$.

    Additional keyword arguments are passed to the matplotlib add_patch function,
    e.g. 'fc' for face colour.
    """
    obs = flavio.classes.Observable[obs_name]
    if not obs.arguments or len(obs.arguments) != 2:
        raise ValueError(r"Only observables that depend on the two bin boundaries (and nothing else) are allowed")
    if wc is None:
        wc = flavio.physics.eft._wc_sm # SM Wilson coefficients
        obs_dict = {bin_: flavio.sm_prediction(obs_name, *bin_) for bin_ in bin_list}
        obs_err_dict = {bin_: flavio.sm_uncertainty(obs_name, *bin_, N=N, threads=threads) for bin_ in bin_list}
    else:
        obs_dict = {bin_:flavio.np_prediction(obs_name, wc, *bin_) for bin_ in bin_list}
        obs_err_dict = {bin_: flavio.np_uncertainty(obs_name, wc, *bin_, N=N, threads=threads) for bin_ in bin_list}
    ax = plt.gca()
    for _i, (bin_, central_) in enumerate(obs_dict.items()):
        xmin, xmax = bin_
        err = obs_err_dict[bin_]
        if divide_binwidth:
            err = err/(xmax-xmin)
            central = central_/(xmax-xmin)
        else:
            central = central_
        if 'fc' not in kwargs and 'facecolor' not in kwargs:
            kwargs['fc'] = 'C6'
        if 'linewidth' not in kwargs and 'lw' not in kwargs:
            kwargs['lw'] = 0
        if _i > 0:
            # the label should only be set for one (i.e. the first)
            # of the boxes, otherwise it will appear multiply in the legend
            kwargs.pop('label', None)
        ax.add_patch(patches.Rectangle((xmin, central-err), xmax-xmin, 2*err,**kwargs))

def bin_plot_exp(obs_name, col_dict=None, divide_binwidth=False, include_measurements=None,
                include_bins=None, exclude_bins=None,
                scale_factor=1,
                **kwargs):
    r"""Plot all existing binned experimental measurements of an observable
    dependending on a continuous parameter, e.g. $q^2$ (in the form of
    coloured crosses).

    Parameters:

    - `col_dict` (optional): a dictionary to assign colours to specific
      experiments, e.g. `{'BaBar': 'b', 'Belle': 'r'}`
    - `divide_binwidth` (optional): this should be set to True when comparing
      integrated branching ratios from experiments with different bin widths
      or to theory predictions for a differential branching ratio. It will
      divide all values and uncertainties by the bin width (i.e. dimensionless
      integrated BRs will be converted to $q^2$-integrated differential BRs with
      dimensions of GeV$^{-2}$). Defaults to False.
    - `include_measurements` (optional): a list of strings with measurement
      names (see measurements.yml) to include in the plot. By default, all
      existing measurements will be included.
    - `include_bins` (optional): a list of bins (as tuples of the bin
      boundaries) to include in the plot. By default, all measured bins
      will be included. Should not be specified simultaneously with
      `exclude_bins`.
    - `exclude_bins` (optional): a list of bins (as tuples of the bin
      boundaries) not to include in the plot. By default, all measured bins
      will be included. Should not be specified simultaneously with
      `include_bins`.
    - `scale_factor` (optional): factor by which all values will be multiplied.
      Defaults to 1.

    Additional keyword arguments are passed to the matplotlib errorbar function,
    e.g. 'c' for colour.
    """
    obs = flavio.classes.Observable[obs_name]
    if not obs.arguments or len(obs.arguments) != 2:
        raise ValueError(r"Only observables that depend on the two bin boundaries (and nothing else) are allowed")
    _experiment_labels = [] # list of experiments appearing in the plot legend
    bins = []
    for m_name, m_obj in flavio.Measurement.instances.items():
        if include_measurements is not None and m_name not in include_measurements:
            continue
        obs_name_list = m_obj.all_parameters
        obs_name_list_binned = [o for o in obs_name_list if isinstance(o, tuple) and o[0]==obs_name]
        if not obs_name_list_binned:
            continue
        central = m_obj.get_central_all()
        err = m_obj.get_1d_errors_rightleft()
        x = []
        y = []
        dx = []
        dy_lower = []
        dy_upper = []
        for _, xmin, xmax in obs_name_list_binned:
            if include_bins is not None:
                if exclude_bins is not None:
                    raise ValueError("Please only specify include_bins or exclude_bins, not both")
                elif (xmin, xmax) not in include_bins:
                    continue
            elif exclude_bins is not None:
                if (xmin, xmax) in exclude_bins:
                    continue
            bins.append((xmin, xmax))
            c = central[(obs_name, xmin, xmax)]
            e_right, e_left = err[(obs_name, xmin, xmax)]
            if divide_binwidth:
                c = c/(xmax-xmin)
                e_left = e_left/(xmax-xmin)
                e_right = e_right/(xmax-xmin)
            ax=plt.gca()
            x.append((xmax+xmin)/2.)
            dx.append((xmax-xmin)/2)
            y.append(c)
            dy_lower.append(e_left)
            dy_upper.append(e_right)
        kwargs_m = kwargs.copy() # copy valid for this measurement only
        if x or y: # only if a data point exists
            if col_dict is not None:
                if m_obj.experiment in col_dict:
                    col = col_dict[m_obj.experiment]
                    kwargs_m['c'] = col
            if 'label' not in kwargs_m:
                if m_obj.experiment not in _experiment_labels:
                    # if there is no plot legend entry for the experiment yet,
                    # add it and add the experiment to the list keeping track
                    # of existing labels (we don't want an experiment to appear
                    # twice in the legend)
                    kwargs_m['label'] = m_obj.experiment
                    _experiment_labels.append(m_obj.experiment)
            y = scale_factor * np.array(y)
            dy_lower = scale_factor * np.array(dy_lower)
            dy_upper = scale_factor * np.array(dy_upper)
            ax.errorbar(x, y, yerr=[dy_lower, dy_upper], xerr=dx, fmt='.', **kwargs_m)
    return y, bins


def diff_plot_exp(obs_name, col_dict=None, include_measurements=None,
                include_x=None, exclude_x=None,
                scale_factor=1,
                **kwargs):
    r"""Plot all existing experimental measurements of an observable
    dependending on a continuous parameter, e.g. $q^2$ (in the form of
    coloured error bars).

    Parameters:

    - `col_dict` (optional): a dictionary to assign colours to specific
      experiments, e.g. `{'BaBar': 'b', 'Belle': 'r'}`
    - `include_measurements` (optional): a list of strings with measurement
      names (see measurements.yml) to include in the plot. By default, all
      existing measurements will be included.
    - `include_x` (optional): a list of values
      to include in the plot. By default, all measured values
      will be included. Should not be specified simultaneously with
      `exclude_x`.
    - `exclude_x` (optional): a list of values
      not to include in the plot. By default, all measured values
      will be included. Should not be specified simultaneously with
      `include_x`.
    - `scale_factor` (optional): factor by which all values will be multiplied.
      Defaults to 1.

    Additional keyword arguments are passed to the matplotlib errorbar function,
    e.g. 'c' for colour.
    """
    obs = flavio.classes.Observable[obs_name]
    if not obs.arguments or len(obs.arguments) != 1:
        raise ValueError(r"Only observables that depend on a single variable are allowed")
    _experiment_labels = [] # list of experiments appearing in the plot legend
    xs = []
    for m_name, m_obj in flavio.Measurement.instances.items():
        if include_measurements is not None and m_name not in include_measurements:
            continue
        obs_name_list = m_obj.all_parameters
        obs_name_list_x = [o for o in obs_name_list if isinstance(o, tuple) and o[0]==obs_name]
        if not obs_name_list_x:
            continue
        central = m_obj.get_central_all()
        err = m_obj.get_1d_errors_rightleft()
        x = []
        y = []
        dy_lower = []
        dy_upper = []
        for _, X in obs_name_list_x:
            if include_x is not None:
                if exclude_x is not None:
                    raise ValueError("Please only specify include_x or exclude_x, not both")
                elif X not in include_x:
                    continue
            elif exclude_x is not None:
                if X in exclude_x:
                    continue
            xs.append(X)
            c = central[(obs_name, X)]
            e_right, e_left = err[(obs_name, X)]
            ax=plt.gca()
            x.append(X)
            y.append(c)
            dy_lower.append(e_left)
            dy_upper.append(e_right)
        kwargs_m = kwargs.copy() # copy valid for this measurement only
        if x or y: # only if a data point exists
            if col_dict is not None:
                if m_obj.experiment in col_dict:
                    col = col_dict[m_obj.experiment]
                    kwargs_m['c'] = col
            if 'label' not in kwargs_m:
                if m_obj.experiment not in _experiment_labels:
                    # if there is no plot legend entry for the experiment yet,
                    # add it and add the experiment to the list keeping track
                    # of existing labels (we don't want an experiment to appear
                    # twice in the legend)
                    kwargs_m['label'] = m_obj.experiment
                    _experiment_labels.append(m_obj.experiment)
            y = scale_factor * np.array(y)
            dy_lower = scale_factor * np.array(dy_lower)
            dy_upper = scale_factor * np.array(dy_upper)
            ax.errorbar(x, y, yerr=[dy_lower, dy_upper], fmt='.', **kwargs_m)
    return y, xs


def density_contour_data(x, y, covariance_factor=None, n_bins=None, n_sigma=(1, 2)):
    r"""Generate the data for a plot with confidence contours of the density
    of points (useful for MCMC analyses).

    Parameters:

    - `x`, `y`: lists or numpy arrays with the x and y coordinates of the points
    - `covariance_factor`: optional, numerical factor to tweak the smoothness
    of the contours. If not specified, estimated using Scott's/Silverman's rule.
    The factor should be between 0 and 1; larger values means more smoothing is
    applied.
    - n_bins: number of bins in the histogram created as an intermediate step.
      this usually does not have to be changed.
    - n_sigma: integer or iterable of integers specifying the contours
      corresponding to the number of sigmas to be drawn. For instance, the
      default (1, 2) draws the contours containing approximately 68 and 95%
      of the points, respectively.
    """
    if n_bins is None:
        n_bins = min(10*int(sqrt(len(x))), 200)
    f_binned, x_edges, y_edges = np.histogram2d(x, y, density=True, bins=n_bins)
    x_centers = (x_edges[:-1] + x_edges[1:])/2.
    y_centers = (y_edges[:-1] + y_edges[1:])/2.
    x_mean = np.mean(x_centers)
    y_mean = np.mean(y_centers)
    dataset = np.vstack([x, y])

    d = 2 # no. of dimensions

    if covariance_factor is None:
        # Scott's/Silverman's rule
        n = len(x) # no. of data points
        _covariance_factor = n**(-1/6.)
    else:
        _covariance_factor = covariance_factor

    cov = np.cov(dataset) * _covariance_factor**2
    gaussian_kernel = scipy.stats.multivariate_normal(mean=[x_mean, y_mean], cov=cov)

    x_grid, y_grid = np.meshgrid(x_centers, y_centers)
    xy_grid = np.vstack([x_grid.ravel(), y_grid.ravel()])
    f_gauss = gaussian_kernel.pdf(xy_grid.T)
    f_gauss = np.reshape(f_gauss, (len(x_centers), len(y_centers))).T

    f = scipy.signal.fftconvolve(f_binned, f_gauss, mode='same').T
    f = f/f.sum()

    def find_confidence_interval(x, pdf, confidence_level):
        return pdf[pdf > x].sum() - confidence_level
    def get_level(n):
        return scipy.optimize.brentq(find_confidence_interval, 0., 1.,
                                     args=(f.T, confidence_level(n)))
    if isinstance(n_sigma, Number):
        levels = [get_level(n_sigma)]
    else:
        levels = [get_level(m) for m in sorted(n_sigma)]

    # replace negative or zero values by a tiny number before taking the log
    f[f <= 0] = 1e-32
    # convert probability to -2*log(probability), i.e. a chi^2
    f = -2*np.log(f)
    # convert levels to chi^2 and make the mode equal chi^2=0
    levels = list(-2*np.log(levels) - np.min(f))
    f = f - np.min(f)

    return {'x': x_grid, 'y': y_grid, 'z': f, 'levels': levels}


def density_contour(x, y, covariance_factor=None, n_bins=None, n_sigma=(1, 2),
                    **kwargs):
    r"""A plot with confidence contours of the density of points
    (useful for MCMC analyses).

    Parameters:

    - `x`, `y`: lists or numpy arrays with the x and y coordinates of the points
    - `covariance_factor`: optional, numerical factor to tweak the smoothness
    of the contours. If not specified, estimated using Scott's/Silverman's rule.
    The factor should be between 0 and 1; larger values means more smoothing is
    applied.
    - n_bins: number of bins in the histogram created as an intermediate step.
      this usually does not have to be changed.
    - n_sigma: integer or iterable of integers specifying the contours
      corresponding to the number of sigmas to be drawn. For instance, the
      default (1, 2) draws the contours containing approximately 68 and 95%
      of the points, respectively.

    All remaining keyword arguments are passed to the `contour` function
    and allow to control the presentation of the plot (see docstring of
    `flavio.plots.plotfunctions.contour`).
    """
    data = density_contour_data(x=x, y=y, covariance_factor=covariance_factor,
                                n_bins=n_bins, n_sigma=n_sigma)
    data['z_min'] = np.min(data['z']) # set minimum to prevent warning
    data.update(kwargs) #  since we cannot do **data, **kwargs in Python <3.5
    return contour(**data)


def likelihood_contour_data(log_likelihood, x_min, x_max, y_min, y_max,
              n_sigma=1, steps=20, threads=1, pool=None):
    r"""Generate data required to plot coloured confidence contours (or bands)
    given a log likelihood function.

    Parameters:

    - `log_likelihood`: function returning the logarithm of the likelihood.
      Can e.g. be the method of the same name of a FastFit instance.
    - `x_min`, `x_max`, `y_min`, `y_max`: data boundaries
    - `n_sigma`: plot confidence level corresponding to this number of standard
      deviations. Either a number (defaults to 1) or a tuple to plot several
      contours.
    - `steps`: number of grid steps in each dimension (total computing time is
      this number squared times the computing time of one `log_likelihood` call!)
    - `threads`: number of threads, defaults to 1. If greater than one,
      computation of z values will be done in parallel.
    - `pool`: an instance of `multiprocessing.Pool` (or a compatible
    implementation, e.g. from `multiprocess` or `schwimmbad`). Overrides the
    `threads` argument.
    """
    _x = np.linspace(x_min, x_max, steps)
    _y = np.linspace(y_min, y_max, steps)
    x, y = np.meshgrid(_x, _y)
    if threads == 1:
        @np.vectorize
        def chi2_vect(x, y): # needed for evaluation on meshgrid
            return -2*log_likelihood([x,y])
        z = chi2_vect(x, y)
    else:
        xy = np.array([x, y]).reshape(2, steps**2).T
        pool = pool or Pool(threads)
        try:
            z = -2*np.array(pool.map(log_likelihood, xy )).reshape((steps, steps))
        except PicklingError:
            pool.close()
            raise PicklingError("When using more than 1 thread, the "
                                "log_likelihood function must be picklable; "
                                "in particular, you cannot use lambda expressions.")
        pool.close()
        pool.join()

    # get the correct values for 2D confidence/credibility contours for n sigma
    if isinstance(n_sigma, Number):
        levels = [delta_chi2(n_sigma, dof=2)]
    else:
        levels = [delta_chi2(n, dof=2) for n in n_sigma]
    return {'x': x, 'y': y, 'z': z, 'levels': levels}


def likelihood_contour(log_likelihood, x_min, x_max, y_min, y_max,
              n_sigma=1, steps=20, threads=1,
              **kwargs):
    r"""Plot coloured confidence contours (or bands) given a log likelihood
    function.

    Parameters:

    - `log_likelihood`: function returning the logarithm of the likelihood.
      Can e.g. be the method of the same name of a FastFit instance.
    - `x_min`, `x_max`, `y_min`, `y_max`: data boundaries
    - `n_sigma`: plot confidence level corresponding to this number of standard
      deviations. Either a number (defaults to 1) or a tuple to plot several
      contours.
    - `steps`: number of grid steps in each dimension (total computing time is
      this number squared times the computing time of one `log_likelihood` call!)

    All remaining keyword arguments are passed to the `contour` function
    and allow to control the presentation of the plot (see docstring of
    `flavio.plots.plotfunctions.contour`).
    """
    data = likelihood_contour_data(log_likelihood=log_likelihood,
                                x_min=x_min, x_max=x_max,
                                y_min=y_min, y_max=y_max,
                                n_sigma=n_sigma, steps=steps, threads=threads)
    data.update(kwargs) #  since we cannot do **data, **kwargs in Python <3.5
    return contour(**data)

# alias for backward compatibility
def band_plot(log_likelihood, x_min, x_max, y_min, y_max,
              n_sigma=1, steps=20, **kwargs):
    r"""This is an alias for `likelihood_contour` which is present for
    backward compatibility."""
    warnings.warn("The `band_plot` function has been replaced "
                  "by `likelihood_contour` (or "
                  "`likelihood_contour_data` in conjunction with `contour`) "
                  "and might be removed in the future. "
                  "Please update your code.", FutureWarning)
    valid_args = inspect.signature(likelihood_contour_data).parameters.keys()
    data_kwargs = {k:v for k,v in kwargs.items() if k in valid_args}
    if 'pre_calculated_z' not in kwargs:
        contour_kwargs = likelihood_contour_data(log_likelihood,
                      x_min, x_max, y_min, y_max,
                      n_sigma, steps, **data_kwargs)
    else:
        contour_kwargs = {}
        nx, ny = kwargs['pre_calculated_z'].shape
        _x = np.linspace(x_min, x_max, nx)
        _y = np.linspace(y_min, y_max, ny)
        x, y = np.meshgrid(_x, _y)
        contour_kwargs['x'] = x
        contour_kwargs['y'] = y
        contour_kwargs['z'] = kwargs['pre_calculated_z']
        if isinstance(n_sigma, Number):
            contour_kwargs['levels'] = [delta_chi2(n_sigma, dof=2)]
        else:
            contour_kwargs['levels'] = [delta_chi2(n, dof=2) for n in n_sigma]
    valid_args = inspect.signature(contour).parameters.keys()
    contour_kwargs.update({k:v for k,v in kwargs.items() if k in valid_args})
    contour(**contour_kwargs)
    return contour_kwargs['x'], contour_kwargs['y'], contour_kwargs['z']


def contour(x, y, z, levels, *, z_min=None,
              interpolation_factor=1,
              interpolation_order=2,
              col=None, color=None, label=None,
              filled=True,
              contour_args={}, contourf_args={},
              **kwargs):
    r"""Plot coloured confidence contours (or bands) given numerical input
    arrays.

    Parameters:

    - `x`, `y`: 2D arrays containg x and y values as returned by numpy.meshgrid
    - `z` value of the function to plot. 2D array in the same shape as `x` and
      `y`.
    - levels: list of function values where to draw the contours. They should
      be positive and in ascending order.
    - `z_min` (optional): lowest value of the function to plot (i.e. value at
      the best fit point). If not provided, the smallest value on the grid is
      used.
    - `interpolation factor` (optional): in between the points on the grid,
      the functioncan be interpolated to get smoother contours.
      This parameter sets the number of subdivisions (default: 1, i.e. no
      interpolation). It should be larger than 1.
    - `col` (optional): number between 0 and 9 to choose the color of the plot
      from a predefined palette
    - `label` (optional): label that will be added to a legend created with
       maplotlib.pyplot.legend()
    - `filled` (optional): if False, contours will be drawn without shading
    - `contour_args`: dictionary of additional options that will be passed
       to matplotlib.pyplot.contour() (that draws the contour lines)
    - `contourf_args`: dictionary of additional options that will be passed
       to matplotlib.pyplot.contourf() (that paints the contour filling).
       Ignored if `filled` is false.
    """
    if z_min is None:
        warnings.warn("The smallest `z` value on the grid will be used as the "
                      "minimum of the function to plot. This can lead to "
                      "undesired results if the actual minimum is considerably "
                      "different from the minimum on the grid. For better "
                      "precision, the actual minimum should be provided in the "
                      "`z_min` argument.")
        z_min = np.min(z) # use minmum on the grid
    elif np.min(z) < z_min:
        raise ValueError("The provided minimum `z_min` has to be smaller than "
                         "the smallest `z` value on the grid.")
    z = z - z_min # subtract z minimum to make value of new z minimum 0
    if interpolation_factor > 1:
        x = scipy.ndimage.zoom(x, zoom=interpolation_factor, order=1)
        y = scipy.ndimage.zoom(y, zoom=interpolation_factor, order=1)
        z = scipy.ndimage.zoom(z, zoom=interpolation_factor, order=interpolation_order)
    _contour_args = {}
    _contourf_args = {}
    color = get_color(col=col, color=color)
    _contour_args['colors'] = color
    if filled:
        _contour_args['linewidths'] = 0.6
    else:
        _contour_args['linewidths'] = 0.8
    N = len(levels)
    _contourf_args['colors'] = [lighten_color(color, 0.5)# RGB
                                       + (max(1-n/N, 0),) # alpha, decreasing for contours
                                       for n in range(N)]
    _contour_args['linestyles'] = 'solid'
    _contour_args.update(contour_args)
    _contourf_args.update(contourf_args)
    # for the filling, need to add zero contour
    zero_contour = min(np.min(z),np.min(levels)*(1-1e-16))
    levelsf = [zero_contour] + list(levels)
    ax = plt.gca()
    if filled:
        ax.contourf(x, y, z, levels=levelsf, **_contourf_args)
    CS = ax.contour(x, y, z, levels=levels, **_contour_args)
    if label is not None:
        CS.collections[0].set_label(label)
    return CS


def flavio_branding(x=0.8, y=0.94, version=True):
    """Displays a little box containing 'flavio'"""
    props = dict(facecolor='white', alpha=0.4, lw=0)
    ax = plt.gca()
    text = r'\textsf{\textbf{flavio}}'
    if version:
        text += r'\textsf{\scriptsize{ v' + flavio.__version__ + '}}'
    ax.text(x, y, text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props, alpha=0.4)

def flavio_box(x_min, x_max, y_min, y_max):
    ax = plt.gca()
    ax.add_patch(patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, facecolor='#ffffff', edgecolor='#666666', alpha=0.5, ls=':', lw=0.7))

def smooth_histogram(data, bandwidth=None, **kwargs):
    """A smooth histogram based on a Gaussian kernel density estimate.

    Parameters:

    - `data`: input array
    - `bandwidth`: (optional) smoothing bandwidth for the Gaussian kernel

    The remaining parameters will be passed to `pdf_plot`.
    """
    kde = flavio.statistics.probability.GaussianKDE(data, bandwidth=bandwidth)
    pdf_plot(kde, **kwargs)

def pdf_plot(dist, x_min=None, x_max=None, fill=True, steps=500, normed=True, **kwargs):
    """Plot of a 1D probability density function.

    Parameters:

    - `dist`: an instance of ProbabilityDistribution
    - `x_min`, `x_max`: plot boundaries
    - `steps`: optional, number of points (default: 200)

    The remaining parameters will be passed to `likelihood_plot`.
    """
    _x_min = x_min or dist.support[0]
    _x_max = x_max or dist.support[1]
    x = np.linspace(_x_min, _x_max, steps)
    try:
        y = dist.pdf(x)
    except:
        y = np.exp(dist.logpdf(x))
    if normed == 'max':
        y = y/np.max(y)
    if fill:
        fill_left = dist.central_value - dist.get_error_left(method='hpd')
        fill_right = dist.central_value + dist.get_error_right(method='hpd')
        fill_x=[fill_left, fill_right]
    else:
        fill_x=None
    likelihood_plot(x, y, fill_x=fill_x, **kwargs)

def likelihood_plot(x, y, fill_x=None, col=None, color=None, label=None, plotargs={}, fillargs={},
                    flipped=False):
    """Plot of a 1D probability density function.

    Parameters:

    - `x`: x values
    - `y`: y values
    - `fill_x`: 2-tuple of x-values in between which the curve will be filled
    - `col`: (optional) integer to select one of the colours from the default
      palette
    - `plotargs`: keyword arguments passed to the `plot` function
    - `fillargs`: keyword arguments passed to the `fill_between` function
    - `flipped`: exchange x and y axes (needed for `density_contour_joint`)
    """
    ax = plt.gca()
    _plotargs = {}
    _fillargs = {}
    # default values
    _plotargs['linewidth'] = 0.6
    if label is not None:
        _plotargs['label'] = label
    color = get_color(col=col, color=color)
    _plotargs['color'] = color
    _fillargs['facecolor'] = lighten_color(color, 0.5)
    _fillargs.update(fillargs)
    _plotargs.update(plotargs)
    if not flipped:
        ax.plot(x, y, **_plotargs)
        if fill_x is not None:
            ax.fill_between(x, 0, y,
                where=np.logical_and(fill_x[0] < x, x < fill_x[1]),
                **_fillargs)
    else:
        ax.plot(y, x, **_plotargs)
        if fill_x is not None:
            ax.fill_betweenx(x, 0, y,
                where=np.logical_and(fill_x[0] < x, x < fill_x[1]),
                **_fillargs)

def pvalue_plot(x, y, fill_y=None, col=None, color=None, label=None,
                plotargs={}, fillargs={}):
    """Plot of a 1D confidence level distribution, where the y axis is 1-CL.

    Parameters:

    - `x`: x values
    - `y`: y values
    - `fill_y`: for x-values where y is larger than this number, the area
      between the x-axis and the curve will be filled
    - `col`: (optional) integer to select one of the colours from the default
      palette
    - `plotargs`: keyword arguments passed to the `plot` function
    - `fillargs`: keyword arguments passed to the `fill_between` function
    """
    ax = plt.gca()
    _plotargs = {}
    _fillargs = {}
    # default values
    _plotargs['linewidth'] = 0.6
    if label is not None:
        _plotargs['label'] = label
    color = get_color(col=col, color=color)
    _plotargs['color'] = color
    _fillargs['facecolor'] = lighten_color(color, 0.5)
    _fillargs.update(fillargs)
    _plotargs.update(plotargs)
    ax.plot(x, y, **_plotargs)
    if fill_y is not None:
        x_zoom = scipy.ndimage.zoom(x, zoom=10, order=1)
        y_zoom = scipy.ndimage.zoom(y, zoom=10, order=1)
        ax.fill_between(x_zoom, 0, y_zoom,
            where=y_zoom > fill_y,
            **_fillargs)
    ax.set_ylim([0, 1])

def density_contour_joint(x, y,
                          col=None, color=None,
                          bandwidth_x=None, bandwidth_y=None,
                          hist_args=None,
                          ax_2d=None, ax_x=None, ax_y=None,
                          **kwargs):
    r"""A density contour plot that additionally has the 1D marginals for
    the x and y dsitribution plotted as smooth histograms along the axes.

    Parameters:

    - `x`, `y`: lists or numpy arrays with the x and y coordinates of the points
    - `covariance_factor`: optional, numerical factor to tweak the smoothness
       of the 2D contours (see `density_contour_data`)
    - `col`: optional, integer specifying the colour, will be applied to both
      contour plot and marginals
    - `bandwidth_x`: optional, smoothing bandwidth for the Gaussian kernel of the
      x marginal distribution
    - `bandwidth_y`: optional, smoothing bandwidth for the Gaussian kernel of the
      y marginal distribution

    Additional options can be passed as follows:

    - `hist_args`: dictionary with keyword arguments passed to the 1D
      `smooth_histogram` for both the x and y distribution
    - Additional keyword arguments will be passed to `density_contour`

    To plot multiple distributions in one figure, the function returns a
    dictionary with the three axis instances that can then be passed into
    another call of the function, e.g.

    ```
    axes1 = density_contour_joint(x1, y1, col=0)
    axes2 = density_contour_joint(x2, y2, col=1, **axes1)
    ```
    """
    # define the plot grid
    gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios = [1, 4])
    gs.update(hspace=0, wspace=0)
    # set up axes (unless they are already given)
    if ax_2d is None:
        ax_2d = plt.subplot(gs[1, 0])
    # make the 2D density contour plot
    density_contour(x, y, col=col, color=color, **kwargs)

    # x axis histogram
    if hist_args is None:
        hist_args = {}
    if ax_x is None:
        ax_x = plt.subplot(gs[0, 0], sharex=ax_2d, yticks=[], frameon=False)
    plt.sca(ax_x)
    smooth_histogram(x, bandwidth=bandwidth_x, col=col, color=color, **hist_args)

    # y axis histogram
    if ax_y is None:
        ax_y = plt.subplot(gs[1, 1], sharey=ax_2d, xticks=[], frameon=False)
    plt.sca(ax_y)
    smooth_histogram(y, flipped=True, bandwidth=bandwidth_y, col=col, color=color, **hist_args)

    # remove x and y histogram tick labels
    plt.setp(ax_x.get_xticklabels(), visible=False);
    plt.setp(ax_y.get_yticklabels(), visible=False);

    # set 2D plot as active axis for adding legends etc.
    plt.sca(ax_2d)
    plt.tight_layout()
    return {'ax_2d': ax_2d, 'ax_x': ax_x, 'ax_y': ax_y}
