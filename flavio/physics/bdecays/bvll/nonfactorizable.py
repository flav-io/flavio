r"""Functions for spectator scattering corrections to $B\to V\ell^+\ell^-$ decays.

This includes weak annihilation, chromomagnetic contributions, and light
quark-loop spectator scattering.
"""

import flavio
import numpy as np
from flavio.classes import AuxiliaryQuantity, Implementation
from flavio.physics.bdecays.common import meson_quark
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict
from flavio.physics.common import conjugate_par, conjugate_wc, add_dict
from flavio.config import config



# Auxiliary quantities and implementations

# function needed for the QCD factorization implementation (see qcdf.py)
def ha_qcdf_function(B, V):
    scale = config['renormalization scale']['bvll']
    label = meson_quark[(B,V)] + 'ee' # the lepton flavour is irrelevant here
                                      # as only dipole and 4-quark operators contribute!
    def function(wc_obj, par_dict, q2, cp_conjugate):
        par = par_dict.copy()
        if cp_conjugate:
            par = conjugate_par(par)
        wc = wctot_dict(wc_obj, label, scale, par)
        if cp_conjugate:
            wc = conjugate_wc(wc)
        return flavio.physics.bdecays.bvll.qcdf.helicity_amps_qcdf(q2, wc, par, B, V)
    return function

# ... and the same for the interpolated version (see qcdf_interpolate.py)
def ha_qcdf_interpolate_function(B, V, contribution='all'):
    scale = config['renormalization scale']['bvll']
    def function(wc_obj, par_dict, q2, cp_conjugate):
        return flavio.physics.bdecays.bvll.qcdf_interpolate.helicity_amps_qcdf(q2, par_dict, B, V, cp_conjugate, contribution)
    return function

# loop over hadronic transitions and lepton flavours
# BTW, it is not necessary to loop over tau: for tautau final states, the minimum
# q2=4*mtau**2 is so high that QCDF is not valid anymore anyway!
for had in [('B0','K*0'), ('B+','K*+'), ('B0','rho0'), ('B+','rho+'), ('Bs','phi'), ]:
    process = had[0] + '->' + had[1] + 'll' # e.g. B0->K*0mumu
    quantity = process + ' spectator scattering'
    a = AuxiliaryQuantity(name=quantity, arguments=['q2', 'cp_conjugate'])
    a.description = ('Contribution to ' + process + ' helicity amplitudes from'
                    ' non-factorizable spectator scattering.')

    # Implementation: QCD factorization
    iname = process + ' QCDF'
    i = Implementation(name=iname, quantity=quantity,
                   function=ha_qcdf_function(B=had[0], V=had[1]))
    i.set_description("QCD factorization")

    # Implementation: interpolated QCD factorization
    iname = process + ' QCDF interpolated'
    i = Implementation(name=iname, quantity=quantity,
                   function=ha_qcdf_interpolate_function(B=had[0], V=had[1]))
    i.set_description("Interpolated version of QCD factorization")
