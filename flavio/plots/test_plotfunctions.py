import unittest
import matplotlib
# matplotlib.use('Agg')
import flavio
import flavio.plots
import numpy as np

# NB, this test only runs with matplotlib>=1.5.3 due to a matplotlib bug

class TestPlots(unittest.TestCase):

    def test_band_plot(self):
        def dummy_loglikelihood(x):
            return -x[0]**2-x[1]**2
        # check that no error is raised and output dimensions match
        x, y, z = flavio.plots.band_plot(dummy_loglikelihood,
                                         -2, 2, -3, 3, steps=30)
        self.assertEqual(x.shape, (30, 30))
        self.assertEqual(y.shape, (30, 30))
        self.assertEqual(z.shape, (30, 30))
        # with interpolation_factor
        x, y, z = flavio.plots.band_plot(dummy_loglikelihood,
                                         -2, 2, -3, 3, steps=30,
                                         interpolation_factor=2)
        self.assertEqual(x.shape, (60, 60))
        self.assertEqual(y.shape, (60, 60))
        self.assertEqual(z.shape, (60, 60))
        # with pre_calculated_z
        x, y, z = flavio.plots.band_plot(None, -2, 2, -3, 3,
                                         pre_calculated_z=z,
                                         interpolation_factor=2)
        # note we now interpolated twice - which is not a good idea in practice.
        self.assertEqual(x.shape, (120, 120))
        self.assertEqual(y.shape, (120, 120))
        self.assertEqual(z.shape, (120, 120))
