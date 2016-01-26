from math import sqrt,pi,log
import pkgutil
import numpy as np
from io import StringIO
import scipy.interpolate
from flavio.physics.running import running
from flavio.physics import ckm


# SM Wilson coefficients at 120 GeV in the basis
# [ C_1, C_2, C_3, C_4, C_5, C_6,
# C_7^eff, C_8^eff,
# C_9, C_10,
# C_3^Q, C_4^Q, C_5^Q, C_6^Q,
# Cb ]
# where all operators are defined as in hep-ph/0512066 *except*
# C_9,10, which are defined with an additional alpha/4pi prefactor.
_wcsm_120 = np.zeros(34)
_wcsm_120[:15] = np.array([  1.99030910e-01,   1.00285703e+00,  -4.17672471e-04,
         2.00964137e-03,   5.20961618e-05,   9.65703651e-05,
        -1.98510105e-01,  -1.09453204e-01,   1.52918563e+00,
        -4.06926405e+00,   6.15944332e-03,   0.00000000e+00,
        -1.12876870e-03,   0.00000000e+00,  -3.24099235e-03])

def wctot_dict(wc_obj, sector, scale, par):
    r"""Get a dictionary with the total (SM + new physics) values  of the
    $\Delta F=1$ Wilson coefficients at a given scale, given a
    WilsonCoefficients instance."""
    wc_np = wc_obj.get_wc(sector, scale, par)
    wc_sm = running.get_wilson(par, _wcsm_120, wc_obj.rge_derivative[sector], 120., scale)
    wc_labels = wc_obj.coefficients[sector]
    wc_dict =  dict(zip(wc_labels, wc_np + wc_sm))
    return wc_dict
