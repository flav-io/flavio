import unittest
from .running import *
import numpy as np

par = {}
par['alpha_s'] = 0.1185
par['alpha_e'] = 1/127.
par[('mass','Z')] = 91.1876
par[('mass','b')] = 4.18
par[('mass','d')] = 4.8e-3
par[('mass','s')] = 0.095
par[('mass','t')] = 173.21
par[('mass','c')] = 1.275
par[('mass','u')] = 2.3e-3

par_pdg = par.copy()
par_pdg['alpha_s'] = 0.1193
par_pdg['alpha_e'] = 1/127.94
par_pdg[('mass','Z')] = 91.1876
par_pdg[('mass','b')] = 4.18

# to compare to RunDec
par_noqed = par.copy()
par_noqed['alpha_e'] = 0.

class TestRunning(unittest.TestCase):

    def test_runningcouplings(self):
        alpha_b = get_alpha(par_noqed, 4.2)
        # compare to 3-loop alpha_s at 4.2 GeV according to RunDec
        self.assertAlmostEqual(alpha_b['alpha_s']/0.225911,1.,places=4)
        # compare to alpha_em at m_tau as given on p. 4 of
        # http://pdg.lbl.gov/2015/reviews/rpp2014-rev-standard-model.pdf
        alpha_tau = get_alpha(par_pdg, 1.777)
        self.assertAlmostEqual(1/alpha_tau['alpha_e']/133.465,1.,places=2)

    def test_runningmasses(self):
        mb_mZ = get_mb(par, 91.1876)
        mu_mb = get_mu(par, 4.18)
        md_mb = get_md(par, 4.18)
        ms_mb = get_ms(par, 4.18)
        mc_mb = get_mc(par, 4.18)
        #TODO add RunDec comparison
        print(mb_mZ, mu_mb, md_mb, ms_mb, mc_mb, )
