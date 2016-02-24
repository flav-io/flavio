from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import flavio
from scipy.optimize import brentq
from scipy.stats import gaussian_kde

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

def density_contour(x, y, xmin, xmax, ymin, ymax, covariance_factor=None):
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)

    if covariance_factor is not None:
        kernel.covariance_factor = lambda: covariance_factor
        kernel._compute_covariance()

    f = np.reshape(kernel(positions).T, xx.shape)*(ymax-ymin)*(xmax-xmin)/1e4

    one_sigma = brentq(find_confidence_interval, 0., 1., args=(f.T, 0.68))
    two_sigma = brentq(find_confidence_interval, 0., 1., args=(f.T, 0.95))
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

def q2_plot_th_bin(obs_name, bin_list, wc=None, N=50, **kwargs):
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
    for bin_, central in obs_dict.items():
        q2min, q2max = bin_
        err = obs_err_dict[bin_]
        if 'fc' not in kwargs and 'facecolor' not in kwargs:
            kwargs['fc'] = flavio.plots.colors.pastel[3]
        if 'linewidth' not in kwargs and 'lw' not in kwargs:
            kwargs['lw'] = 0
        ax.add_patch(patches.Rectangle((q2min, central-err), q2max-q2min, 2*err,**kwargs))

def q2_plot_exp(obs_name, col_dict=None, **kwargs):
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
            e = err[(obs_name, q2min, q2max)]
            ax=plt.gca()
            if col_dict is not None:
                if m_obj.experiment in col_dict:
                    col = col_dict[m_obj.experiment]
                    kwargs['c'] = col
            ax.errorbar((q2max+q2min)/2., c, yerr=e, xerr=(q2max-q2min)/2, **kwargs)
