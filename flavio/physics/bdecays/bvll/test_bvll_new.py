import unittest
import numpy as np
from . import angulardist, amplitudes, angular_new
from flavio.physics.bdecays.formfactors.b_v import bsz_parameters
from flavio.physics.eft import WilsonCoefficients
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict

s = 1.519267515435317e+24

par = {
    ('mass','e'): 1e-16,
    ('mass','mu'): 105.6583715e-3,
    ('mass','tau'): 1.77686,
    ('mass','B+'): 5.27929,
    ('mass','B0'): 5.27961,
    ('mass','Bs'): 5.36679,
    ('mass','K*0'): 0.89166,
    ('lifetime','B+'): 1638.e-15*s,
    ('lifetime','B0'): 152.e-14*s,
    'alpha_e': 1/127.940,
    'alpha_s': 0.1185,
    ('mass','Z'): 91.1876,
    ('mass','b'): 4.17,
    ('mass','t'): 173.21,
    ('mass','c'): 1.275,
    'Gmu': 1.1663787e-5,
    'Vus': 0.22,
    'Vub': 3.7e-3,
    'Vcb': 4.1e-2,
    'gamma': 1.22,
    ('f','B0'): 0.1905,
    ('f_perp','K*0'): 0.161,
    ('f_para','K*0'): 0.211,
    ('a1_perp','K*0'): 0.03,
    ('a1_para','K*0'): 0.02,
    ('a2_perp','K*0'): 0.08,
    ('a2_para','K*0'): 0.08,
}

par.update(bsz_parameters.ffpar_lcsr)

wc_obj = WilsonCoefficients()
wc_ini = {
'CS_bsmumu': 1.31,
'CP_bsmumu': -3.25,
'CSp_bsmumu': 1.86j + 3.,
'CPp_bsmumu': -1.38j,
'C7effp_bs': -3.25,
'C9p_bsmumu': 1.86j + 3.,
'C10p_bsmumu': -1.38j,
}
wc_obj.set_initial(wc_ini, 4.8)
wc = wctot_dict(wc_obj, 'bsmumu', 4.8, par)

class TestBVll(unittest.TestCase):
    def test_bksll(self):
        q2 = 1.5
        # compare helicity amplitudes
        a = amplitudes.transversity_amps_ff(q2, wc, par, 'B0', 'K*0', 'mu')
        p = angular_new.prefactor(q2, par, 'B0', 'K*0', 'mu')/amplitudes.prefactor(q2, par, 'B0', 'K*0', 'mu')/4
        a_new = {k: p*v for k, v in a.items()}
        h = angular_new.helicity_amps_ff(q2, wc, par, 'B0', 'K*0', 'mu')
        h2 = angular_new.transversity_to_helicity(a_new)
        for k in h2.keys():
            if h[k] != 0:
                self.assertAlmostEqual(h2[k]/h[k], 1, places=12)
        # compare helicity amplitudes
        a = amplitudes.transversity_amps_qcdf(q2, wc, par, 'B0', 'K*0', 'mu')
        p = angular_new.prefactor(q2, par, 'B0', 'K*0', 'mu')/amplitudes.prefactor(q2, par, 'B0', 'K*0', 'mu')/4
        a_new = {k: p*v for k, v in a.items()}
        h = angular_new.helicity_amps_qcdf(q2, wc, par, 'B0', 'K*0', 'mu')
        h2 = angular_new.transversity_to_helicity(a_new)
        for k in h2.keys():
            if h[k] != 0:
                self.assertAlmostEqual(h2[k]/h[k], 1, places=12)
        # compare angular coefficients
        a = amplitudes.transversity_amps(q2, wc, par, 'B0', 'K*0', 'mu')
        J = angulardist.angulardist(a, q2, par, 'mu')
        J_new = angular_new.get_angularcoeff(q2, wc_obj, par, 'B0', 'K*0', 'mu')
        for k in J:
            if J_new[k] != 0:
                self.assertAlmostEqual(J[k]/(J_new[k]), 1, places=12)
