import unittest
import numpy as np
import flavio

constraints = flavio.default_parameters
wc_obj = flavio.WilsonCoefficients()
par = constraints.get_central_all()

class TestBlnu(unittest.TestCase):
    def test_blnu(self):
        Vub = flavio.physics.ckm.get_ckm(par)[0,2]
        # compare to literature value
        self.assertAlmostEqual(
            flavio.Observable.get_instance("BR(B+->taunu)").prediction_central(constraints, wc_obj),
            1.1e-4 * (abs(Vub)/3.95e-3)**2 * (par['f_B+']/0.2)**2,
            delta=2e-6)
        # check that B->enu BR is smaller than B->munu
        # (ratio given by mass ratio squared)
        self.assertAlmostEqual(
            (
            flavio.Observable.get_instance("BR(B+->enu)").prediction_central(constraints, wc_obj)/
            flavio.Observable.get_instance("BR(B+->munu)").prediction_central(constraints, wc_obj)
            )/(par['m_e']**2/par['m_mu']**2),
            1,
            delta=0.001) # there are corrections of order mmu**2/mB**2
