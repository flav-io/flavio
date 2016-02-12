from collections import OrderedDict
import matplotlib.pyplot as plt
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

def density_contour_kernel(x, y, xmin, xmax, ymin, ymax, covariance_factor=None):
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
    print(levels)

    ax = plt.gca()
    ax.contourf(xx, yy, f, levels=levels, colors=flavio.plots.colors.reds[3:])
    ax.contour(xx, yy, f, levels=levels, colors=flavio.plots.colors.reds[::-1], linewidths=0.7)
