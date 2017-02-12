r"""Module for making pretty plots based on matplotlib.

Currently, the following types of plots are supported:

- Plotting the error budget of a prediction (`error_budget_pie`)
- Plots of experimental measurements and theory predictions of $q^2$ dependent
  observables, such as differential branching ratios
  (`q2_plot_exp`, `q2_plot_th_bin`, `q2_plot_th_diff`)
- Plots for two-dimensional contours (e.g. frequentist confidence contours
  or Bayesian credibility contours) that can be used standalone or in
  conjunction with `flavio.statistics.Fit` instances to plot the fit results
  (`density_contour`, `likelihood_contour`, `contours`)
- Plots for one-dimensional likelihoods and confidence/credibility intervals
  (`smooth_histogram`)
"""

from . import colors
from .plotfunctions import *
from . import config
