"""Functions for interpolated version of the QCDF spectator scattering
corrections.

Rationale for this is that
a) The QCDF corrections are very small, in fact smaller than hadronic uncertainties
for most observables (except maybe isospin asymmetries)
b) They are very costly computing-wise as they involve numerical integrals
over light-cone distribution amplitudes.

Interpolating them leads to a drastic speed-up at the cost of some (unnecessary)
precision.
"""

import flavio
import numpy as np
import pkgutil
import pkg_resources
import scipy.interpolate
import math
import warnings

q2_arr = np.arange(0 + 1e-6, 9 + 1e-6, 0.1)

data = np.load(pkg_resources.resource_filename('flavio.physics', 'data/qcdf_interpolate/qcdf_interpolate.npz'))

interpolating_function_dict = {}
for process, hel_amps in data.items():
    interpolating_function_dict[process] = {}
    interpolating_function_dict[process] = scipy.interpolate.interp1d(q2_arr, hel_amps.view(float), axis=0)

def helicity_amps_qcdf(q2, par, B, V, cp_conjugate=False, contribution='all'):
    if q2 > 6:
        warnings.warn("The QCDF corrections should not be trusted for q2 above 6 GeV^2")
    # ml = par['m_'+lep]
    # if lep == 'tau' or q2 < 4*ml**2 or q2 > 9:
    #     return {('0' ,'V'): 0, ('pl' ,'V'): 0, ('mi' ,'V'): 0}
    process = B + '->' + V # e.g. B0->K*0mumu
    if cp_conjugate:
        array_name = process + ' CP conjugate ' + contribution
    else:
        array_name = process + ' ' + contribution
    ha = {}
    ha_arr = interpolating_function_dict[array_name](q2).view(complex)
    # dividing the q^2 poles back in
    ha[('pl','V')] = ha_arr[0] / q2
    ha[('mi','V')] = ha_arr[1] / q2
    ha[('0','V')]  = ha_arr[2] / math.sqrt(q2)
    return ha
