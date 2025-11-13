import cmath
from flavio.classes import AuxiliaryQuantity, Implementation
from .. import angular
from . import amplitudes
from ... import mesonmixing
from . import amplitudes_transversity


def angular_coefficients_helicity(wc_obj, par, q2, ff, B, V, lep, ml, mB, mV, mb, corrections=True):
    """
    Returns the angular coefficients J, J_bar, J_h, J_s for the B->Vll decay
    in the helicity amplitude formalism.
    """
    h = amplitudes.helicity_amps(q2, ff, wc_obj, par, B, V, lep, corrections=corrections)
    h_bar = amplitudes.helicity_amps_bar(q2, ff, wc_obj, par, B, V, lep, corrections=corrections)
    J = angular.angularcoeffs_general_v(h, q2, mB, mV, mb, 0, ml, ml)
    J_bar = angular.angularcoeffs_general_v(h_bar, q2, mB, mV, mb, 0, ml, ml)
    h_tilde = h_bar.copy()
    h_tilde[('pl', 'V')] = h_bar[('mi', 'V')]
    h_tilde[('pl', 'A')] = h_bar[('mi', 'A')]
    h_tilde[('mi', 'V')] = h_bar[('pl', 'V')]
    h_tilde[('mi', 'A')] = h_bar[('pl', 'A')]
    h_tilde['S'] = -h_bar['S']
    q_over_p = mesonmixing.observables.q_over_p(wc_obj, par, B)
    phi = cmath.phase(-q_over_p) # the phase of -q/p
    J_h = angular.angularcoeffs_h_v(phi, h, h_tilde, q2, mB, mV, mb, 0, ml, ml)
    J_s = angular.angularcoeffs_s_v(phi, h, h_tilde, q2, mB, mV, mb, 0, ml, ml)
    return J, J_bar, J_h, J_s


def angular_coefficients_transversity(wc_obj, par, q2, ff, B, V, lep, ml, mB, mV, mb):
    """ 
    Returns the angular coefficients J, J_bar, J_h, J_s for the B->Vll decay
    in the transversity amplitude formalism.
    """
    A = amplitudes_transversity.transversity_amps(q2, ff, wc_obj, par, B, V, lep)
    A_bar = amplitudes_transversity.transversity_amps_bar(q2, ff, wc_obj, par, B, V, lep)
    J = amplitudes_transversity.angularcoeffs_general_transversity(A, q2, ml)
    J_bar = amplitudes_transversity.angularcoeffs_general_transversity(A_bar, q2, ml)
    A_tilde = A_bar.copy()
    A_tilde['perp_L'] = -1 * A_bar['perp_L']
    A_tilde['perp_R'] = -1 * A_bar['perp_R']
    A_tilde['S'] = -1 * A_bar['S']  # Table 3 of https://arxiv.org/pdf/1502.05509
    q_over_p = mesonmixing.observables.q_over_p(wc_obj, par, B)
    J_h: dict[str | int, float] = amplitudes_transversity.angularcoeffs_h_transversity(A, A_tilde, q2, ml, q_over_p)
    J_s: dict[str | int, float] = amplitudes_transversity.angularcoeffs_s_transversity(A, A_tilde, q2, ml, q_over_p)
    return J, J_bar, J_h, J_s


quantity = 'B->Vll amplitude formalism'
a = AuxiliaryQuantity(name=quantity, arguments=['q2', 'ff', 'B', 'V', 'lep', 'ml', 'mB', 'mV', 'mb'])
a.set_description('J, J_bar, J_h, J_s coefficients for the B->Vll decay amplitude.')

iname = 'Helicity Amplitudes'
i = Implementation(name=iname, quantity=quantity,
                   function=lambda wc_obj, par, q2, ff, B, V, lep, ml, mB, mV, mb: angular_coefficients_helicity(wc_obj, par, q2, ff, B, V, lep, ml, mB, mV, mb, corrections=True))
i.set_description("Angular coefficients from the helicity amplitude formalism, including all corrections from subleading effects and QCDF.")

iname = 'Helicity Amplitudes (no corrections)'
i = Implementation(name=iname, quantity=quantity,
                   function=lambda wc_obj, par, q2, ff, B, V, lep, ml, mB, mV, mb: angular_coefficients_helicity(wc_obj, par, q2, ff, B, V, lep, ml, mB, mV, mb, corrections=False))
i.set_description("Angular coefficients from the helicity amplitude formalism, without corrections from subleading effects and QCDF.")

iname = 'Transversity Amplitudes (no corrections)'
i = Implementation(name=iname, quantity=quantity,
                   function=angular_coefficients_transversity)
i.set_description("Angular coefficients from the transversity amplitude formalism, without corrections from subleading effects and QCDF.")
