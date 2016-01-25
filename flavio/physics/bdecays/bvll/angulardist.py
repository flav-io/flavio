from math import sqrt,pi
import numpy as np
from flavio.physics.bdecays.common import lambda_K, beta_l, meson_quark, meson_ff
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict
from flavio.physics.bdecays import matrixelements
from flavio.physics import ckm
from flavio.physics.bdecays.formfactors import FormFactorParametrization as FF
from flavio.config import config
from flavio.physics.running import running

"""Functions for exclusive $B\to V\ell^+\ell^-$ decays."""


def angulardist(transversity_amps, q2, par, lep):
    r"""Returns the angular coefficients of the 4-fold differential decay
    distribution of a $B\to V\ell^+\ell^-$ decay as defined in eq. (3.9)-(3.11)
    of arXiv:0811.1214.

    Input:
      - transversity_amps: dictionary containing the transverity amplitudes
      - q2: dilepton invariant mass squared $q_2$ in GeV$^2$
      - par: parameter dictionary
      - lep: lepton flavour, either 'e', 'mu', or 'tau'
    """
    ml = par[('mass',lep)]
    ApaL = transversity_amps['para_L']
    ApaR = transversity_amps['para_R']
    ApeL = transversity_amps['perp_L']
    ApeR = transversity_amps['perp_R']
    A0L = transversity_amps['0_L']
    A0R = transversity_amps['0_R']
    AS = transversity_amps['S']
    At = transversity_amps['t']
    J = {}
    J['1s'] = (1/(4 * q2) * ((-4 * ml**2 + 3 * q2) * abs(ApaL)**2
     + (-4 * ml**2 + 3 * q2) * abs(ApaR)**2 - 4 * ml**2 * abs(ApeL)**2 + 3 * q2 * abs(ApeL)**2 -
     4 * ml**2 * abs(ApeR)**2 + 3 * q2 * abs(ApeR)**2 +
     16 * ml**2 * np.real(ApaL * np.conj(ApaR)) +
     16 * ml**2 * np.real(ApeL * np.conj(ApeR))))
    J['1c'] = (abs(A0L)**2 + abs(A0R)**2 + (4 * ml**2 * abs(At)**2)/q2
     + beta_l(ml, q2)**2 * abs(AS)**2 + (8 * ml**2 * np.real(A0L * np.conj(A0R)))/q2)
    J['2s'] = (1/4 * beta_l(ml, q2)**2 * (abs(ApaL)**2 + abs(ApaR)**2 + abs(ApeL)**2 + abs(ApeR)**2))
    J['2c'] = (-beta_l(ml, q2)**2 * (abs(A0L)**2 + abs(A0R)**2))
    J[3] = (1/2 * beta_l(ml, q2)**2 * (-abs(ApaL)**2 - abs(ApaR)**2 + abs(ApeL)**2 + abs(ApeR)**2))
    J[4] = ((beta_l(ml, q2)**2 * (np.real(A0L * np.conj(ApaL)) + np.real(A0R * np.conj(ApaR))))/sqrt(2))
    J[5] = (sqrt(2) * beta_l(ml, q2) * (np.real(A0L * np.conj(ApeL)) - np.real(A0R * np.conj(ApeR)) -
     ml /sqrt(q2) * (np.real(ApaL * np.conj(AS)) + np.real(ApaR * np.conj(AS)))))
    J['6s'] = (2 * beta_l(ml, q2) * (np.real(ApaL * np.conj(ApeL)) - np.real(ApaR * np.conj(ApeR))))
    J['6c'] = (4 * ml/sqrt(q2) * beta_l(ml, q2) * (np.real(A0L * np.conj(AS)) + np.real(A0R * np.conj(AS))))
    J[7] = (sqrt(2) * beta_l(ml, q2) * (np.imag(A0L * np.conj(ApaL)) - np.imag(A0R * np.conj(ApaR)) +
     ml/sqrt(q2) * (np.imag(ApeL * np.conj(AS)) + np.imag(ApeR * np.conj(AS)))))
    J[8] = ((beta_l(ml, q2)**2 * (np.imag(A0L * np.conj(ApeL)) + np.imag(A0R * np.conj(ApeR))))/sqrt(2))
    J[9] = (-beta_l(ml, q2)**2 * (np.imag(ApaL * np.conj(ApeL)) + np.imag(ApaR * np.conj(ApeR))))
    return J

def angulardist_bar(transversity_amps_bar, q2, par, lep):
    J = angulardist(transversity_amps_bar, q2, par, lep)
    J[5] = -J[5]
    J['6s'] = -J['6s']
    J['6c'] = -J['6c']
    J[8] = -J[8]
    J[9] = -J[9]
    return J
