import unittest
import numpy.testing as npt
import matplotlib
import flavio
from flavio.plots import *
import numpy as np
import scipy.stats
import warnings

from matplotlib import rc
# (to avoid tex errors on Travis CI)
rc('text', usetex=False)
rc('font',**{'family':'sans-serif'})

def dummy_loglikelihood(x):
    return -x[0]**2-x[1]**2

class TestPlots(unittest.TestCase):
    def test_error_budget_pie(self):
        err_budget_bsmumu = {'DeltaGamma/Gamma_Bs': 0.0048858044281356464,
             'GF': 1.1082519567930236e-06,
             'Vcb': 0.039790267942911725,
             'Vub': 0.00039636600693301931,
             'Vus': 0.00040185451336679405,
             'alpha_e': 0.00017640953398991146,
             'alpha_s': 1.7980573055638197e-05,
             'f_Bs': 0.035073683628645283,
             'delta': 0.0040950620820486205,
             'm_Bs': 4.395718017004336e-05,
             'm_mu': 3.89716785433222e-08,
             'tau_Bs': 0.003286868163723475}
        error_budget_pie(err_budget_bsmumu)

    def test_q2_th_diff(self):
        # without specifying WCs
        diff_plot_th('dBR/dq2(B0->pienu)', 0, 25, steps=10, scale_factor=1000)
        # with WCs
        diff_plot_th('dBR/dq2(B+->pienu)', 0, 25,
                                   wc=flavio.WilsonCoefficients(), steps=10)
        # check that observable not depending on q2 raises error
        with self.assertRaises(ValueError):
            diff_plot_th('eps_K', 0, 25)

    def test_q2_th_diff_err(self):
        # without parallelization
        diff_plot_th_err('dBR/dq2(B0->pienu)', 1, 24, steps=5,
                                                steps_err=3, N=10,
                                                scale_factor=3)
        # with parallelization
        diff_plot_th_err('dBR/dq2(B0->pienu)', 1, 24, steps=5,
                                                 steps_err=3, N=10, threads=2)

    def test_q2_th_bin(self):
        bins = [(0, 5), (5, 10)]
        # without specifying WCs
        bin_plot_th('<BR>(B0->pienu)', bins, N=10)
        # with WCs
        bin_plot_th('<BR>(B+->pienu)', bins, divide_binwidth=True,
                                          wc=flavio.WilsonCoefficients(), N=10)
        # check that observable not depending on q2 raises error
        with self.assertRaises(ValueError):
            bin_plot_th('eps_K', bins)

    def test_q2_plot_exp(self):
        # vanilla
        bin_plot_exp('<dBR/dq2>(B0->K*mumu)')
        # with options
        bin_plot_exp('<dBR/dq2>(B0->K*mumu)', col_dict={'LHCb': 'r'},
                                                        exclude_bins=[(1.1, 6)],
                                                        scale_factor=100)
        # check that observable not depending on q2 raises error
        with self.assertRaises(ValueError):
            bin_plot_exp('eps_K')

    def test_q2_plot_exp(self):
        # vanilla
        m = flavio.Measurement('test measurement diff_plot_exp')
        m.set_constraint(('dBR/dq2(B0->K*mumu)', 1), '1 +- 0.1 e-6')
        m.set_constraint(('dBR/dq2(B0->K*mumu)', 2), '2 +- 0.2 e-6')
        diff_plot_exp('dBR/dq2(B0->K*mumu)')
        diff_plot_exp('dBR/dq2(B0->K*mumu)', scale_factor=10)
        # with options
        # check that observable not depending on q2 raises error
        with self.assertRaises(ValueError):
            diff_plot_exp('eps_K')
        # remove test measurement
        del flavio.Measurement['test measurement diff_plot_exp']

    def test_band_plot(self):
        # NB, this test only runs with matplotlib>=1.5.3 due to a matplotlib bug
        # check that no error is raised and output dimensions match
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            x, y, z = band_plot(dummy_loglikelihood,
                                             -2, 2, -3, 3, steps=30)
            self.assertEqual(x.shape, (30, 30))
            self.assertEqual(y.shape, (30, 30))
            self.assertEqual(z.shape, (30, 30))
            # with interpolation_factor
            x, y, z = band_plot(dummy_loglikelihood,
                                             -2, 2, -3, 3, steps=30,
                                             interpolation_factor=2)
            self.assertEqual(x.shape, (30, 30))
            self.assertEqual(y.shape, (30, 30))
            self.assertEqual(z.shape, (30, 30))
            # with pre_calculated_z
            x, y, z = band_plot(None, -2, 2, -3, 3,
                                             pre_calculated_z=z,
                                             interpolation_factor=2)
            self.assertEqual(x.shape, (30, 30))
            self.assertEqual(y.shape, (30, 30))
            self.assertEqual(z.shape, (30, 30))

    def test_density_contour_data(self):
        np.random.seed(42)
        xy = scipy.stats.multivariate_normal(mean=[2,3], cov=[[1,0.5],[0.5,1]]).rvs(size=100)
        data = density_contour_data(*xy.T)
        self.assertEqual(data['x'].shape, (100,100))
        self.assertEqual(data['y'].shape, (100,100))
        self.assertEqual(data['z'].shape, (100,100))
        self.assertEqual(len(data['levels']), 2) # by default 2 levels (1, 2sigma)
        self.assertTrue(min(data['levels']) > 0) # levels positive
        self.assertEqual(data['levels'], sorted(data['levels'])) # levels ascending
        self.assertEqual(np.min(data['z']), 0)
        # point in the middle should be close to maximum likelihood
        self.assertAlmostEqual(data['z'][50,50], 0, delta=0.1)
        # corners
        self.assertTrue(data['z'][0,0] < data['z'][-1,0])
        self.assertTrue(data['z'][-1,-1] < data['z'][0,-1])
        # symmetries
        self.assertAlmostEqual(data['z'][-1,-1], data['z'][0,0], delta=1.)
        self.assertAlmostEqual(data['z'][-1,0], data['z'][0,-1], delta=3.)

    def test_density_contour(self):
        # just check this works
        np.random.seed(42)
        xy = scipy.stats.multivariate_normal(mean=[2,3], cov=[[1,0.5],[0.5,1]]).rvs(size=100)
        density_contour(*xy.T)
        density_contour_joint(*xy.T)

    def test_likelihood_contour(self):
        # just check this works
        data = likelihood_contour_data(dummy_loglikelihood,
                                        -2, 2, -3, 3)
        self.assertEqual(data['x'].shape, (20,20))
        self.assertEqual(data['y'].shape, (20,20))
        self.assertEqual(data['z'].shape, (20,20))
        self.assertEqual(len(data['levels']), 1) # by default, plot 1 sigma contour
        self.assertAlmostEqual(data['levels'][0], 2.3, delta=0.01) #
        self.assertAlmostEqual(np.min(data['z']), 0.07202216) # keep value of mininum
        # test parallel computation
        data2 = likelihood_contour_data(dummy_loglikelihood,
                                        -2, 2, -3, 3, threads=2)
        npt.assert_array_equal(data2['z'], data['z'])
        # check that `z_min` larger than `np.min(z)` raises error
        with self.assertRaises(ValueError):
            kwargs = {'z_min':0.1}
            kwargs.update(data) #  since we cannot do **data, **kwargs in Python <3.5
            contour(**kwargs)

    def test_smooth_histogram(self):
        # just check this doesn't raise and error
        np.random.seed(42)
        dat = np.random.normal(117, 23, size=100)
        smooth_histogram(dat, col=1)
