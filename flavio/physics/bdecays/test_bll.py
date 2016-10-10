import unittest
from .bll import *
import numpy as np
from .. import ckm
from math import radians
from flavio.physics.eft import WilsonCoefficients
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict
from flavio.classes import Parameter, Observable
from flavio.parameters import default_parameters
import copy
import flavio

s = 1.519267515435317e+24

c = copy.deepcopy(default_parameters)
# parameters taken from PDG and table I of 1311.0903
c.set_constraint('alpha_s', '0.1184(7)')
c.set_constraint('f_Bs', '0.2277(45)')
c.set_constraint('f_B0', '0.1905(42)')
c.set_constraint('Vcb', 4.24e-2)
c.set_constraint('Vub', 3.82e-3)
c.set_constraint('gamma', radians(73.))
c.set_constraint('DeltaGamma/Gamma_Bs', 0.1226)

par = c.get_central_all()


wc_obj = WilsonCoefficients()
wc = wctot_dict(wc_obj, 'bsmumu', 4.8, par)
wc_e = wctot_dict(wc_obj, 'bsee', 4.8, par)
wc_tau = wctot_dict(wc_obj, 'bstautau', 4.8, par)


class TestBll(unittest.TestCase):
    def test_bsll(self):
        # just some trivial tests to see if calling the functions raises an error
        self.assertGreater(br_lifetime_corr(0.08, -1), 0)
        self.assertEqual(len(amplitudes(par, wc, 'Bs', 'mu', 'mu')), 2)
        # ADeltaGamma should be +1.0 in the SM
        self.assertEqual(ADeltaGamma(par, wc, 'Bs', 'mu'), 1.0)
        # BR should be around 3.5e-9
        self.assertAlmostEqual(br_inst(par, wc, 'Bs', 'mu', 'mu')*1e9, 3.5, places=0)
        # correction factor should enhance the BR by roughly 7%
        self.assertAlmostEqual(br_timeint(par, wc, 'Bs', 'mu', 'mu')/br_inst(par, wc, 'Bs', 'mu', 'mu'), 1.07, places=2)
        # ratio of Bs->mumu and Bs->ee BRs should be roughly given by ratio of squared masses
        self.assertAlmostEqual(
            br_timeint(par, wc_e, 'Bs', 'e', 'e')/br_timeint(par, wc, 'Bs', 'mu', 'mu')/par['m_e']**2*par['m_mu']**2,
            1., places=2)
        # comparison to 1311.0903
        self.assertAlmostEqual(abs(ckm.xi('t','bs')(par))/par['Vcb'], 0.980, places=3)
        self.assertAlmostEqual(br_timeint(par, wc, 'Bs', 'mu', 'mu')/3.65e-9, 1, places=1)
        self.assertAlmostEqual(br_timeint(par, wc_e, 'Bs', 'e', 'e')/8.54e-14, 1, places=1)
        self.assertAlmostEqual(br_timeint(par, wc_tau, 'Bs', 'tau', 'tau')/7.73e-7, 1, places=1)

    def test_bsll_classes(self):
        par_default = default_parameters.get_central_all()
        self.assertAlmostEqual(br_timeint(par_default, wc_tau, 'Bs', 'tau', 'tau')/Observable.get_instance('BR(Bs->tautau)').prediction_central(default_parameters, wc_obj), 1, places=4)
        self.assertAlmostEqual(br_timeint(par_default, wc_e, 'Bs', 'e', 'e')/Observable.get_instance('BR(Bs->ee)').prediction_central(default_parameters, wc_obj), 1, places=4)
        self.assertAlmostEqual(br_timeint(par_default, wc, 'Bs', 'mu', 'mu')/Observable.get_instance('BR(Bs->mumu)').prediction_central(default_parameters, wc_obj), 1, places=4)

    def test_bsll_lfv(self):
        # test for errors
        self.assertEqual(flavio.sm_prediction('BR(B0->emu)'), 0)
        self.assertEqual(flavio.sm_prediction('BR(Bs->taumu)'), 0)

    def test_EffectiveLifetimes(self):
        # In this test we trivially check that the prefactors in (22) and (28) of arXiv:1204.1737 are the same

        ys     = .5*par['DeltaGamma/Gamma_Bs']
        tau_Bs = par['tau_Bs']

        wc_dict = {'e': wc_e, 'mu': wc, 'tau': wc_tau}

        for l in ['e', 'mu', 'tau']:
            ADG    = ADeltaGamma(par, wc_dict[l], 'Bs', l)
            tau    = tau_ll(wc_dict[l], par, 'Bs', l)

            prefactor1 = br_lifetime_corr(ys, ADG)        # eq. (22) of arXiv:1204.1737
            prefactor2 = 2.  - (1.-ys**2) * tau / tau_Bs  # eq. (28) of arXiv:1204.1737

            self.assertAlmostEqual(prefactor1, prefactor2, places=8)
