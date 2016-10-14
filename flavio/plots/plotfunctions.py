from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import flavio
import scipy.optimize
import scipy.interpolate
import scipy.stats
from numbers import Number

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
    err_tot = sum(err_dict.values())
    err_dict_sorted = OrderedDict(sorted(err_dict.items(), key=lambda t: -t[1]))
    labels = []
    fracs = []
    small_frac = 0
    for key, value in err_dict_sorted.items():
        frac = value/err_tot
        if frac > other_cutoff:
            labels.append(flavio.Parameter.get_instance(key).tex)
            fracs.append(frac)
        else:
            small_frac += frac
    if small_frac > 0:
        labels.append('other')
        fracs.append(small_frac)
    def my_autopct(pct):
        return r'{p:.1f}\%'.format(p=pct*err_tot)
    plt.axis('equal')
    return plt.pie(fracs, labels=labels, autopct=my_autopct, wedgeprops = {'linewidth':0.5}, colors=flavio.plots.colors.pastel)


def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level

def density_contour(x, y, covariance_factor=None):
    r"""A contour plot with 1 and 2 $\sigma$ contours of the density of points
    (useful for MCMC analyses).

    Parameters:

    - `x`, `y`: lists or numpy arrays with the x and y coordinates of the points
    - `covariance_factor`: optional, numerical factor to tweak the smoothness
    of the contours
    """
    xmin = min(x)
    ymin = min(y)
    xmax = max(x)
    ymax = max(y)
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = scipy.stats.gaussian_kde(values)

    if covariance_factor is not None:
        kernel.covariance_factor = lambda: covariance_factor
        kernel._compute_covariance()

    f = np.reshape(kernel(positions).T, xx.shape)*(ymax-ymin)*(xmax-xmin)/1e4

    one_sigma = scipy.optimize.brentq(find_confidence_interval, 0., 1., args=(f.T, 0.68))
    two_sigma = scipy.optimize.brentq(find_confidence_interval, 0., 1., args=(f.T, 0.95))
    levels = [two_sigma, one_sigma, np.max(f)]

    ax = plt.gca()
    ax.contourf(xx, yy, f, levels=levels, colors=flavio.plots.colors.reds[3:])
    ax.contour(xx, yy, f, levels=levels, colors=flavio.plots.colors.reds[::-1], linewidths=0.7)


def q2_plot_th_diff(obs_name, q2min, q2max, wc=None, q2steps=100, **kwargs):
    r"""Plot the central theory prediction of a $q^2$-dependent observable
    as a function of $q^2$.

    Parameters:

    - `q2min`, `q2max`: minimum and maximum $q^2$ values in GeV^2
    - `wc` (optional): `WilsonCoefficient` instance to define beyond-the-SM
      Wilson coefficients
    - `q2steps` (optional): number of $q^2$ steps. Defaults to 100. Less is
      faster but less precise.

    Additional keyword arguments are passed to the matplotlib plot function,
    e.g. 'c' for colour.
    """
    obs = flavio.classes.Observable.get_instance(obs_name)
    if obs.arguments != ['q2']:
        raise ValueError(r"Only observables that depend on $q^2$ (and nothing else) are allowed")
    q2_arr = np.arange(q2min, q2max, (q2max-q2min)/(q2steps-1))
    if wc is None:
        wc = flavio.WilsonCoefficients() # SM Wilson coefficients
        obs_arr = [flavio.sm_prediction(obs_name, q2) for q2 in q2_arr]
    else:
        obs_arr = [flavio.np_prediction(obs_name, wc, q2) for q2 in q2_arr]
    ax = plt.gca()
    if 'c' not in kwargs and 'color' not in kwargs:
        kwargs['c'] = 'k'
    ax.plot(q2_arr, obs_arr, **kwargs)

def q2_plot_th_bin(obs_name, bin_list, wc=None, divide_binwidth=False, N=50, **kwargs):
    r"""Plot the binned theory prediction with uncertainties of a
    $q^2$-dependent observable as a function of $q^2$  (in the form of coloured
    boxes).

    Parameters:

    - `bin_list`: a list of tuples containing bin boundaries
    - `wc` (optional): `WilsonCoefficient` instance to define beyond-the-SM
      Wilson coefficients
    - `divide_binwidth` (optional): this should be set to True when comparing
      integrated branching ratios from experiments with different bin widths
      or to theory predictions for a differential branching ratio. It will
      divide all values and uncertainties by the bin width (i.e. dimensionless
      integrated BRs will be converted to integrated differential BRs with
      dimensions of GeV$^{-2}$). Defaults to False.
    - `N` (optional): number of random draws to determine the uncertainty.
      Defaults to 50. Larger is slower but more precise. The relative
      error of the theory uncertainty scales as $1/\sqrt{2N}$.

    Additional keyword arguments are passed to the matplotlib add_patch function,
    e.g. 'fc' for face colour.
    """
    obs = flavio.classes.Observable.get_instance(obs_name)
    if obs.arguments != ['q2min', 'q2max']:
        raise ValueError(r"Only observables that depend on q2min and q2max (and nothing else) are allowed")
    if wc is None:
        wc = flavio.WilsonCoefficients() # SM Wilson coefficients
        obs_dict = {bin_: flavio.sm_prediction(obs_name, *bin_) for bin_ in bin_list}
        obs_err_dict = {bin_: flavio.sm_uncertainty(obs_name, *bin_, N=N) for bin_ in bin_list}
    else:
        obs_dict = {bin_:flavio.np_prediction(obs_name, wc, *bin_) for bin_ in bin_list}
    ax = plt.gca()
    for _i, (bin_, central_) in enumerate(obs_dict.items()):
        q2min, q2max = bin_
        err = obs_err_dict[bin_]
        if divide_binwidth:
            err = err/(q2max-q2min)
            central = central_/(q2max-q2min)
        else:
            central = central_
        if 'fc' not in kwargs and 'facecolor' not in kwargs:
            kwargs['fc'] = flavio.plots.colors.pastel[3]
        if 'linewidth' not in kwargs and 'lw' not in kwargs:
            kwargs['lw'] = 0
        if _i > 0:
            # the label should only be set for one (i.e. the first)
            # of the boxes, otherwise it will appear multiply in the legend
            kwargs.pop('label', None)
        ax.add_patch(patches.Rectangle((q2min, central-err), q2max-q2min, 2*err,**kwargs))

def q2_plot_exp(obs_name, col_dict=None, divide_binwidth=False, include_measurements=None, **kwargs):
    r"""Plot all existing experimental measurements of a $q^2$-dependent
    observable as a function of $q^2$  (in the form of coloured crosses).

    Parameters:

    - `col_dict` (optional): a dictionary to assign colours to specific
      experiments, e.g. `{'BaBar': 'b', 'Belle': 'r'}`
    - `divide_binwidth` (optional): this should be set to True when comparing
      integrated branching ratios from experiments with different bin widths
      or to theory predictions for a differential branching ratio. It will
      divide all values and uncertainties by the bin width (i.e. dimensionless
      integrated BRs will be converted to integrated differential BRs with
      dimensions of GeV$^{-2}$). Defaults to False.
    - `include_measurements` (optional): a list of strings with measurement
      names (see measurements.yml) to include in the plot. By default, all
      existing measurements will be included.

    Additional keyword arguments are passed to the matplotlib errorbar function,
    e.g. 'c' for colour.
    """
    obs = flavio.classes.Observable.get_instance(obs_name)
    if obs.arguments != ['q2min', 'q2max']:
        raise ValueError(r"Only observables that depend on q2min and q2max (and nothing else) are allowed")
    for m_name, m_obj in flavio.Measurement.instances.items():
        if include_measurements is not None and m_name not in include_measurements:
            continue
        obs_name_list = m_obj.all_parameters
        obs_name_list_binned = [o for o in obs_name_list if isinstance(o, tuple) and o[0]==obs_name]
        if not obs_name_list_binned:
            continue
        central = m_obj.get_central_all()
        err = m_obj.get_1d_errors()
        x = []
        y = []
        dx = []
        dy = []
        for _, q2min, q2max in obs_name_list_binned:
            c = central[(obs_name, q2min, q2max)]
            e = err[(obs_name, q2min, q2max)]
            if divide_binwidth:
                c = c/(q2max-q2min)
                e = e/(q2max-q2min)
            ax=plt.gca()
            x.append((q2max+q2min)/2.)
            dx.append((q2max-q2min)/2)
            y.append(c)
            dy.append(e)
        if col_dict is not None:
            if m_obj.experiment in col_dict:
                col = col_dict[m_obj.experiment]
                kwargs['c'] = col
        ax.errorbar(x, y, yerr=dy, xerr=dx, label=m_obj.experiment, fmt='.', **kwargs)


def band_plot(log_likelihood, x_min, x_max, y_min, y_max, n_sigma=1, steps=20,
              interpolation_factor=1,
              col=None, label=None,
              pre_calculated_z=None,
              contour_args={}, contourf_args={}):
    r"""Plot coloured confidence contours (or bands) given a log likelihood
    function.

    Parameters:

    - `log_likelihood`: function returning the logarithm of the likelihood.
      Can e.g. be the method of the same name of a FastFit instance.
    - `x_min`, `x_max`, `y_min`, `y_max`: plot boundaries
    - `n_sigma`: plot confidence level corresponding to this number of standard
      deviations. Either a number (defaults to 1) or a tuple to plot several
      contours.
    - `steps`: number of grid steps in each dimension (total computing time is
      this number squared times the computing time of one `log_likelihood` call!)
    - `interpolation factor` (optional): in between the points on the grid
      set by `steps`, the log likelihood can be interpolated to get smoother contours.
      This parameter sets the number of subdivisions (default: 1, i.e. no
      interpolation). It should be larger than 1.
    - `col` (optional): number between 0 and 9 to choose the color of the plot
      from a predefined palette
    - `label` (optional): label that will be added to a legend created with
       maplotlib.pyplot.legend()
    - `pre_calculated_z` (optional): z values for a band plot, previously
       calculated. In this case, no likelihood scan is performed and time can
       be saved.
    - `contour_args`: dictionary of additional options that will be passed
       to matplotlib.pyplot.contour() (that draws the contour lines)
    - `contourf_args`: dictionary of additional options that will be passed
       to matplotlib.pyplot.contourf() (that paints the contour filling)
    """
    ax = plt.gca()
    # coarse grid
    _x = np.linspace(x_min, x_max, steps)
    _y = np.linspace(y_min, y_max, steps)
    x, y = np.meshgrid(_x, _y)
    @np.vectorize
    def chi2_vect(x, y): # needed for evaluation on meshgrid
        return -2*log_likelihood([x,y])
    if pre_calculated_z is not None:
        z = pre_calculated_z
    else:
        z = chi2_vect(x, y)
    # fine grid
    _x = np.linspace(x_min, x_max, steps*interpolation_factor)
    _y = np.linspace(y_min, y_max, steps*interpolation_factor)
    # interpolate z from coarse to fine grid
    z = scipy.ndimage.zoom(z, zoom=interpolation_factor)
    z = z - np.min(z) # subtract the best fit point (on the grid)
    x, y = np.meshgrid(_x, _y)
    if col is not None and isinstance(col, int):
        contourf_args['colors'] = [flavio.plots.colors.pastel[col]]
        contour_args['colors'] = [flavio.plots.colors.set1[col]]
    else:
        if 'colors' not in contourf_args:
            contourf_args['colors'] = [flavio.plots.colors.pastel[0]]
        if 'colors' not in contour_args:
            contour_args['colors'] = [flavio.plots.colors.set1[0]]
    if 'linestyle' not in contour_args:
        contour_args['linestyles'] = 'solid'
    # get the correct values for 2D confidence/credibility contours for n sigma
    chi2_1dof = scipy.stats.chi2(1)
    chi2_2dof = scipy.stats.chi2(2)
    def get_y(n):
        cl_nsigma = chi2_1dof.cdf(n**2) # this is roughly 0.68 for n_sigma=1 etc.
        return chi2_2dof.ppf(cl_nsigma) # this is roughly 2.3 for n_sigma=1 etc.
    if isinstance(n_sigma, Number):
        levels = [get_y(n_sigma)]
    else:
        levels = [get_y(n) for n in n_sigma]
    # for the filling, need to add zero contour
    levelsf = [0] + levels
    ax.contourf(x, y, z, levels=levelsf, **contourf_args)
    CS = ax.contour(x, y, z, levels=levels, **contour_args)
    if label is not None:
        CS.collections[0].set_label(label)
    return x, y, z


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

def smooth_histogram(data, N=20, plotargs={}, fillargs={}):
    """A smooth (interpolated) histogram. N (default: 20) is the number of
    steps."""
    y, binedges = np.histogram(data, bins=N)
    x = 0.5*(binedges[1:]+binedges[:-1])
    f = scipy.interpolate.interp1d(x, y, kind='cubic')
    ax = plt.gca()
    if 'color' not in plotargs:
        plotargs['color'] = flavio.plots.colors.set1[0]
    if 'facecolor' not in fillargs:
        fillargs['facecolor'] = flavio.plots.colors.pastel[0]
    ax.plot(x, y, **plotargs)
    ax.fill_between(x, 0, y, **fillargs)
