from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import flavio
import scipy.optimize
import scipy.interpolate
import scipy.stats

def error_budget_pie(err_dict, other_cutoff=0.03):
    """Pie chart of an observable's error budget."""
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
    obs = flavio.classes.Observable.get_instance(obs_name)
    if obs.arguments != ['q2min', 'q2max']:
        raise ValueError(r"Only observables that depend on q2min and q2max (and nothing else) are allowed")
    if wc is None:
        wc = flavio.WilsonCoefficients() # SM Wilson coefficients
        obs_dict = {bin_: flavio.sm_prediction(obs_name, *bin_) for bin_ in bin_list}
        obs_err_dict = {bin_: flavio.sm_uncertainty(obs_name, *bin_, N=N) for bin_ in bin_list}
    else:
        wc = flavio.WilsonCoefficients() # SM Wilson coefficients
        obs_dict = {bin_:flavio.np_prediction(obs_name, wc, *bin_) for bin_ in bin_list}
    ax = plt.gca()
    for bin_, central_ in obs_dict.items():
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
        ax.add_patch(patches.Rectangle((q2min, central-err), q2max-q2min, 2*err,**kwargs))

def q2_plot_exp(obs_name, col_dict=None, divide_binwidth=False, **kwargs):
    obs = flavio.classes.Observable.get_instance(obs_name)
    if obs.arguments != ['q2min', 'q2max']:
        raise ValueError(r"Only observables that depend on q2min and q2max (and nothing else) are allowed")
    for m_name, m_obj in flavio.Measurement.instances.items():
        obs_name_list = m_obj.all_parameters
        obs_name_list_binned = [o for o in obs_name_list if isinstance(o, tuple) and o[0]==obs_name]
        if not obs_name_list_binned:
            continue
        central = m_obj.get_central_all()
        err = m_obj.get_1d_errors()
        for _, q2min, q2max in obs_name_list_binned:
            c = central[(obs_name, q2min, q2max)]
            if divide_binwidth:
                c = c/(q2max-q2min)
            e = err[(obs_name, q2min, q2max)]
            ax=plt.gca()
            if col_dict is not None:
                if m_obj.experiment in col_dict:
                    col = col_dict[m_obj.experiment]
                    kwargs['c'] = col
            ax.errorbar((q2max+q2min)/2., c, yerr=e, xerr=(q2max-q2min)/2, **kwargs)

def band_plot(likelihood_fct, x_min, x_max, y_min, y_max, n_sigma=1, steps=20, col=None, contour_args={}, contourf_args={}):
    ax = plt.gca()
    _x = np.linspace(x_min, x_max, steps)
    _y = np.linspace(y_min, y_max, steps)
    x, y = np.meshgrid(_x, _y)
    @np.vectorize
    def f_vect(x, y): # needed for evaluation on meshgrid
        return likelihood_fct([x,y])
    z = f_vect(x, y)
    z = z - np.max(z) # subtract the best fit point (on the grid)
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
    cl_nsigma = chi2_1dof.cdf(n_sigma**2) # this is roughly 0.68 for n_sigma=1 etc.
    y_nsigma = chi2_2dof.ppf(cl_nsigma) # this is roughly 2.3 for n_sigma=1 etc.
    ax.contourf(x, y, z, levels=[-y_nsigma, 0], **contourf_args)
    ax.contour(x, y, z, levels=[-y_nsigma], **contour_args)

def flavio_branding(x=0.8, y=0.94, version=True):
    props = dict(facecolor='white', alpha=0.4, lw=1.2)
    ax = plt.gca()
    text = r'\textsf{\textbf{flavio}}'
    if version:
        text += r'\textsf{\scriptsize{ v' + flavio.__version__ + '}}'
    ax.text(x, y, text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props, alpha=0.4)

def flavio_box(x_min, x_max, y_min, y_max):
    ax = plt.gca()
    ax.add_patch(patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, facecolor='#ffffff', edgecolor='#666666', alpha=0.5, ls=':', lw=0.7))

def smooth_density_histogram(data, N=20, plotargs={}, fillargs={}):
    """A smooth (interpolated) density histogram. N (default: 20) is the number
    of steps."""
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
