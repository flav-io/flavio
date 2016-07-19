"""This file contains the functions that allow to regenerate the data files
for the interpolation of the QCDF corrections. It is normally not needed."""

import flavio
import numpy as np
from flavio.physics.bdecays.bvll import qcdf
from flavio.config import config
from flavio.physics.bdecays.common import meson_quark
from flavio.physics.common import conjugate_par

# we will interpolate from 0 to 9 GeV^2 (above, QCDF is not applicable)
# shifted by 10^-6 to avoid the q2=0 singularity but be below the e+e- threshold
q2_arr = np.arange(0 + 1e-6, 9 + 1e-6, 0.1)

# NP contributions to QCDF corrections will be neglected throughout!
wcsm = flavio.WilsonCoefficients()

# central default values are assumed by all parameters

par = flavio.default_parameters.copy()
par_dict = par.get_central_all()
par_dict_cpconj = conjugate_par(par_dict)


array_dict = {}
i = 0
contributions_dict = {
'all': {'include_WA': True,  'include_O8': True,  'include_QSS': True},
'WA':  {'include_WA': True,  'include_O8': False, 'include_QSS': False},
'O8':  {'include_WA': False, 'include_O8': True,  'include_QSS': False},
'QSS': {'include_WA': False, 'include_O8': False, 'include_QSS': True },
}
for had in [('B0','K*0'), ('B+','K*+'), ('Bs','phi')]: #, ('B0','rho0'), ('B+','rho+') ]:
    for cp_conjugate in [False, True]: # compute it for the decay and its CP conjugate
        for contribution_name, contribution_dict in contributions_dict.items():
            l = 'e'
            process = had[0] + '->' + had[1] # e.g. B0->K*0
            print('Computing ' + process + ', contribution: ' + contribution_name + ' (' + str(i+1) + '/' + str(2*3*4) + ')')
            # compute corrections for each q2 value
            scale = config['renormalization scale']['bvll']
            label = meson_quark[had] + l + l # e.g. bsee
            wcsm_dict =  flavio.physics.bdecays.wilsoncoefficients.wctot_dict(wcsm, label, scale, par_dict)
            if cp_conjugate:
                hel_amps_list = [ qcdf.helicity_amps_qcdf(q2, wcsm_dict, par_dict_cpconj, B=had[0], V=had[1], **contribution_dict)
                              for q2 in q2_arr]
            else:
                hel_amps_list = [ qcdf.helicity_amps_qcdf(q2, wcsm_dict, par_dict, B=had[0], V=had[1],  **contribution_dict)
                              for q2 in q2_arr]
            # turn into numpy array
            hel_amps_arr = np.array([
                [hel_amps[('pl','V')],
                 hel_amps[('mi','V')],
                 hel_amps[('0','V')]]
                for hel_amps in hel_amps_list
            ])
            # divide out the q^2 poles to get a saner interpolant
            hel_amps_arr[:,0] = q2_arr * hel_amps_arr[:,0]
            hel_amps_arr[:,1] = q2_arr * hel_amps_arr[:,1]
            hel_amps_arr[:,2] = np.sqrt(q2_arr) * hel_amps_arr[:,2]
            if cp_conjugate:
                array_name = process + ' CP conjugate ' + contribution_name
            else:
                array_name = process + ' ' + contribution_name
            array_dict[array_name] = hel_amps_arr
            i += 1

np.savez('qcdf_interpolate', **array_dict)
