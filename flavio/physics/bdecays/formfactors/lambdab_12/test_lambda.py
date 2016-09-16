import unittest
import flavio
from flavio.classes import Implementation


class TestLambdabFF(unittest.TestCase):

    def test_lambdab_lambda_ff(self):
        ff_obj = Implementation.get_instance('Lambdab->Lambda SSE2')
        par = flavio.default_parameters
        mL = par.get_central('m_Lambda')
        mLb = par.get_central('m_Lambdab')
        q2max = (mLb-mL)**2
        ff_max = ff_obj.get_central(constraints_obj=par, wc_obj=None, q2=q2max)
        ff_0 = ff_obj.get_central(constraints_obj=par, wc_obj=None, q2=0)
        # g_perp(q2_max) = g_+(q2_max)
        self.assertAlmostEqual(ff_max['fAperp'], ff_max['fA0'], delta=0.001)
        # htilde_perp(q2_max) = htilde_+(q2_max)
        self.assertAlmostEqual(ff_max['fT5perp'], ff_max['fT50'], delta=0.001)
        # f_0(0) = f+(0)
        self.assertAlmostEqual(ff_0['fVt'], ff_0['fV0'], delta=0.05) # this constraint is poorly satisfied
        # g_0(0) = g+(0)
        self.assertAlmostEqual(ff_0['fAt'], ff_0['fA0'], delta=0.1) # this constraint is poorly satisfied
