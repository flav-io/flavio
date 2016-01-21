from math import sqrt,pi,log
import pkgutil
import numpy as np
from io import StringIO
import scipy.interpolate
from flavio.physics.running import running
from flavio.physics import ckm



def YC9(q2):
    #FIXME
    return 0.

# NNLO matrix elements of C_1 and C_2 needed for semi-leptonic B decays

_f_string = pkgutil.get_data('flavio.physics', 'data/arXiv-0810-4077v3/f_12_79.dat')
_f_array = np.loadtxt(StringIO(_f_string.decode('utf-8')))
_f_x = _f_array[::51*11,0]
_f_y = _f_array[:51*11:51,1]
_f_z = _f_array[:51,2]
_f_val_17 = _f_array[:,3].reshape(11,11,51) + 1j*_f_array[:,4].reshape(11,11,51)
_f_val_19 = _f_array[:,5].reshape(11,11,51) + 1j*_f_array[:,6].reshape(11,11,51)
_f_val_27 = _f_array[:,7].reshape(11,11,51) + 1j*_f_array[:,8].reshape(11,11,51)
_f_val_29 = _f_array[:,9].reshape(11,11,51) + 1j*_f_array[:,10].reshape(11,11,51)
_F_17 = scipy.interpolate.RegularGridInterpolator((_f_x, _f_y, _f_z), _f_val_17)
_F_19 = scipy.interpolate.RegularGridInterpolator((_f_x, _f_y, _f_z), _f_val_19)
_F_27 = scipy.interpolate.RegularGridInterpolator((_f_x, _f_y, _f_z), _f_val_27)
_F_29 = scipy.interpolate.RegularGridInterpolator((_f_x, _f_y, _f_z), _f_val_29)

def F_17(muh, z, sh):
    """Function $F_1^{(7)}$ as defined in arXiv:0810.4077.

    `muh` is $\hat\mu=\mu/m_b$,
    `z` is $z=m_c^2/m_b^2$,
    `sh` is $\hat s=q^2/m_b^2$.
    """
    return _F_17([muh, z, sh])[0]
def F_19(muh, z, sh):
    """Function $F_1^{(9)}$ as defined in arXiv:0810.4077.

    `muh` is $\hat\mu=\mu/m_b$,
    `z` is $z=m_c^2/m_b^2$,
    `sh` is $\hat s=q^2/m_b^2$.
    """
    return _F_19([muh, z, sh])[0]
def F_27(muh, z, sh):
    """Function $F_2^{(7)}$ as defined in arXiv:0810.4077.

    `muh` is $\hat\mu=\mu/m_b$,
    `z` is $z=m_c^2/m_b^2$,
    `sh` is $\hat s=q^2/m_b^2$.
    """
    return _F_27([muh, z, sh])[0]
def F_29(muh, z, sh):
    """Function $F_2^{(9)}$ as defined in arXiv:0810.4077.

    `muh` is $\hat\mu=\mu/m_b$,
    `z` is $z=m_c^2/m_b^2$,
    `sh` is $\hat s=q^2/m_b^2$.
    """
    return _F_29([muh, z, sh])[0]
