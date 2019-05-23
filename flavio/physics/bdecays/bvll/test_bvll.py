import unittest
import numpy as np
import flavio
from flavio.physics.bdecays.formfactors.b_v import bsz_parameters
from flavio.physics.bdecays.bvll import observables, observables_bs
import cmath
import warnings


class TestBVll(unittest.TestCase):
    def test_compare_to_Davids_old_code(self):
        # this actually works to 3% and better! Still, setting it to 15%
        # here to not give any error in case of future changes to the central value of the code
        delta = 0.15
        self.assertAlmostEqual(flavio.sm_prediction("AFB(B0->K*mumu)", q2=1)/-0.15349451759556973, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("FL(B0->K*mumu)", q2=1)/0.6406629242990924, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("S5(B0->K*mumu)", q2=1)/0.22071783227496475, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("P4p(B0->K*mumu)", q2=1)/0.15140284164390483, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("P5p(B0->K*mumu)", q2=1)/0.5064923738851054, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("AFB(B0->K*mumu)", q2=6)/0.17920578512694066, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("FL(B0->K*mumu)", q2=6)/0.6746725598733865, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("S4(B0->K*mumu)", q2=6)/-0.24476949751293892, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("S5(B0->K*mumu)", q2=6)/-0.3735858481345886, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("P4p(B0->K*mumu)", q2=6)/-0.529132300413707, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("P5p(B0->K*mumu)", q2=6)/-0.807602014278806, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("AFB(B0->K*mumu)", q2=15)/0.406194755197377, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("FL(B0->K*mumu)", q2=15)/0.36807161547568185, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("S3(B0->K*mumu)", q2=15)/-0.14872271241628923, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("S4(B0->K*mumu)", q2=15)/-0.2919191029733921, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("S5(B0->K*mumu)", q2=15)/-0.34293705381921147, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("P4p(B0->K*mumu)", q2=15)/-0.6065016593226283, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("P5p(B0->K*mumu)", q2=15)/-0.7124983944730857, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("dBR/dq2(B0->K*mumu)", q2=1)/5.7131974656129313e-8, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("dBR/dq2(B0->K*mumu)", q2=6)/5.524478678779278e-8, 1, delta=delta)
        self.assertAlmostEqual(flavio.sm_prediction("dBR/dq2(B0->K*mumu)", q2=15)/7.412258847550239e-8, 1, delta=delta)
        # direct CP asymmetry should be close to 0
        self.assertAlmostEqual(flavio.sm_prediction("ACP(B0->K*mumu)", q2=1), 0, delta=0.01)
        self.assertAlmostEqual(flavio.sm_prediction("ACP(B0->K*mumu)", q2=6), 0, delta=0.01)
        self.assertAlmostEqual(flavio.sm_prediction("ACP(B0->K*mumu)", q2=17), 0, delta=0.01)
        self.assertAlmostEqual(flavio.sm_prediction("ACP(B+->K*mumu)", q2=1), 0, delta=0.01)
        self.assertAlmostEqual(flavio.sm_prediction("ACP(B+->K*mumu)", q2=6), 0, delta=0.01)
        self.assertAlmostEqual(flavio.sm_prediction("ACP(B+->K*mumu)", q2=17), 0, delta=0.01)
        # this is just a very rough comparison to the literature
        delta = 0.2
        self.assertAlmostEqual(flavio.sm_prediction("P1(B0->K*mumu)", q2=3), 0, delta=delta)
        delta = 0.1
        self.assertAlmostEqual(flavio.sm_prediction("P1(B0->K*mumu)", q2=17), -0.64, delta=delta)
        delta = 0.1
        self.assertAlmostEqual(flavio.sm_prediction("P2(B0->K*mumu)", q2=6)/(2/3.*0.179/0.325), 1, delta=delta)
        delta = 0.1
        self.assertAlmostEqual(flavio.sm_prediction("P2(B0->K*mumu)", q2=17)/0.37, 1, delta=delta)
        delta = 0.02
        self.assertAlmostEqual(flavio.sm_prediction("P3(B0->K*mumu)", q2=3), 0, delta=delta)
        delta = 0.02
        self.assertAlmostEqual(flavio.sm_prediction("P3(B0->K*mumu)", q2=17), 0, delta=delta)
        # LFU ratio = 1
        self.assertAlmostEqual(flavio.sm_prediction("Rmue(B0->K*ll)", q2=6), 1, delta=1e-2)
        self.assertAlmostEqual(flavio.sm_prediction("Rmue(B+->K*ll)", q2=6), 1, delta=1e-2)
        self.assertAlmostEqual(flavio.sm_prediction("Dmue_P4p(B0->K*ll)", q2=6), 0, delta=1e-6)
        self.assertAlmostEqual(flavio.sm_prediction("Dmue_P5p(B0->K*ll)", q2=3), 0, delta=1e-2)

    def test_obs(self):
        # just test that this doesn't raise
        flavio.sm_prediction("<AFB>(B0->K*mumu)", 2, 3)
        flavio.sm_prediction("<dBR/dq2>(B+->K*mumu)", 2, 3)
        flavio.sm_prediction("<P5p>(B0->K*ee)", 2, 3)
        flavio.sm_prediction("P5p(B0->K*mumu)", 6)
        flavio.sm_prediction("<P1>(B0->K*mumu)", 2, 3)
        flavio.sm_prediction("P1(B0->K*ee)", 5)
        flavio.sm_prediction("<Rmue>(B0->K*ll)", 2, 3)
        flavio.sm_prediction("Rmue(B0->K*ll)", 4)

    def test_unphysical(self):
        # check BR calculation yields zero outside kinematical limits
        self.assertEqual(flavio.sm_prediction("dBR/dq2(B0->K*mumu)", q2=0.01), 0)
        self.assertEqual(flavio.sm_prediction("dBR/dq2(B0->K*mumu)", q2=25), 0)
        # and also *at* kinemetical limits
        par = flavio.default_parameters.get_central_all()
        q2min = 4*par['m_mu']**2
        q2max = (par['m_B0']-par['m_K*0'])**2
        self.assertAlmostEqual(flavio.sm_prediction("dBR/dq2(B0->K*mumu)", q2=q2min), 0, delta=1e-10)
        self.assertAlmostEqual(flavio.sm_prediction("dBR/dq2(B0->K*mumu)", q2=q2max), 0, delta=1e-10)

        # same for angular observables (make sure no division by 0)
        self.assertEqual(flavio.sm_prediction("S5(B0->K*mumu)", q2=0.01), 0)
        self.assertEqual(flavio.sm_prediction("S5(B0->K*mumu)", q2=25), 0)

    def test_bs_timedep(self):
        q2 = 3
        wc_obj = flavio.WilsonCoefficients()
        par = flavio.default_parameters.get_central_all()
        B = 'Bs'
        V = 'phi'
        l = 'mu'
        # a set of parameters with y_s=0!
        par_y0 = par.copy()
        par_y0['DeltaGamma/Gamma_Bs']=0

        # compare without lifetime effect: must be equal!
        self.assertEqual(
            observables.BVll_obs(observables.dGdq2_ave, q2, B, V, l, wc_obj, par_y0)(),
            observables_bs.bsvll_obs(observables_bs.dGdq2_ave_Bs, q2, wc_obj, par_y0, B, V, l))
        self.assertEqual(
            observables.BVll_obs(observables.FL, q2, B, V, l, wc_obj, par_y0)(),
            observables_bs.bsvll_obs( observables_bs.FL_Bs, q2, wc_obj, par_y0, B, V, l))
        for i in [3, 4, 7]: # S3,4,7
            self.assertEqual(
            observables.BVll_obs(lambda J, J_bar: observables.S_experiment(J, J_bar, i), q2, B, V, l, wc_obj, par_y0)(),
            observables_bs.bsvll_obs( lambda y, J, J_bar, J_h: observables_bs.S_experiment_Bs(y, J, J_bar, J_h, i), q2, wc_obj, par_y0, B, V, l))

        # check that the phase phi has the right convention
        q_over_p = flavio.physics.mesonmixing.observables.q_over_p(wc_obj, par, B)
        phi = cmath.phase(-q_over_p) # the phase of q/p
        self.assertAlmostEqual(phi, 0.04, delta=0.01)

        # compare WITH lifetime effect: angular observables must be similar
        delta = 0.01
        self.assertAlmostEqual(
            observables.BVll_obs(     observables.FL,       q2, B, V, l, wc_obj, par)()/
            observables_bs.bsvll_obs( observables_bs.FL_Bs, q2, wc_obj, par, B, V, l),
            1, delta=delta)
        for i in [4, 7]: # S4,7
            self.assertAlmostEqual(
                observables.BVll_obs(     lambda J, J_bar:         observables.S_experiment(J, J_bar, i),               q2, B, V, l, wc_obj, par)()/
                observables_bs.bsvll_obs( lambda y, J, J_bar, J_h: observables_bs.S_experiment_Bs(y, J, J_bar, J_h, i), q2, wc_obj, par, B, V, l),
                1, delta=delta)
        for i in [3]: # S3: look at differnece only
            self.assertAlmostEqual(
                observables.BVll_obs(     lambda J, J_bar:         observables.S_experiment(J, J_bar, i),               q2, B, V, l, wc_obj, par)() -
                observables_bs.bsvll_obs( lambda y, J, J_bar, J_h: observables_bs.S_experiment_Bs(y, J, J_bar, J_h, i), q2, wc_obj, par, B, V, l),
                0, delta=0.01)
        # compare WITH lifetime effect: BR suppressed by ~6%!
        self.assertAlmostEqual(
            observables.BVll_obs(     observables.dGdq2_ave,       q2, B, V, l, wc_obj, par)()/
            observables_bs.bsvll_obs( observables_bs.dGdq2_ave_Bs, q2, wc_obj, par, B, V, l),
            1.06, delta=0.02)

        # and now just check a few observables to see if any errors are raised
        flavio.sm_prediction("FL(Bs->phimumu)", q2=1)
        flavio.sm_prediction("S3(Bs->phimumu)", q2=1)
        flavio.sm_prediction("S4(Bs->phimumu)", q2=1)
        flavio.sm_prediction("S7(Bs->phimumu)", q2=1)

    def test_bvll_integrate_pole(self):
        def f(q2):
            # dummy function with a pole at q2=0
            return 1e-10*(1 + 0.1*q2 + 0.01*q2**2)/q2
        from flavio.math.integrate import nintegrate
        from flavio.physics.bdecays.bvll.observables import nintegrate_pole
        self.assertAlmostEqual(nintegrate_pole(f, 0.1, 10)/nintegrate(f, 0.1, 10, epsrel=0.001),
                               1, delta=0.01)
        self.assertAlmostEqual(nintegrate_pole(f, 0.001, 0.01)/nintegrate(f, 0.001, 0.01, epsrel=0.001),
                               1, delta=0.01)
        self.assertAlmostEqual(nintegrate_pole(f, 0.0005, 2)/nintegrate(f, 0.0005, 2, epsrel=0.0001),
                               1, delta=0.03)
        # try whether it also works with a well-behaved function
        def g(q2):
            return 1e-10*(1 + 0.1*q2 + 0.01*q2**2)
        self.assertAlmostEqual(nintegrate_pole(g, 0.1, 10)/nintegrate(g, 0.1, 10, epsrel=0.001),
                               1, delta=0.01)
        self.assertAlmostEqual(nintegrate_pole(g, 0.001, 0.01)/nintegrate(g, 0.001, 0.01, epsrel=0.001),
                               1, delta=0.01)
        self.assertAlmostEqual(nintegrate_pole(g, 0.0005, 2)/nintegrate(g, 0.0005, 2, epsrel=0.001),
                               1, delta=0.01)

    def test_qcdf_warning(self):
        # computing the BR at q²=8 should warn
        with self.assertWarnsRegex(UserWarning, r"The QCDF corrections should not be trusted .*"):
            flavio.sm_prediction('dBR/dq2(B0->K*mumu)', 8)
        # computing the LFU ratio at q²=8 should not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            flavio.sm_prediction('Rmue(B0->K*ll)', 8)
