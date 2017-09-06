import unittest
import numpy as np
import numpy.testing as npt
import flavio
from flavio.classes import Observable, Measurement, Parameter, ParameterConstraints, Prediction
from flavio.statistics.fits import FrequentistFit
from flavio.statistics.fitters import profiler
import scipy.stats

class TestProfilers(unittest.TestCase):

    def test_shuffle(arg):
        npt.assert_array_equal(profiler.reshuffle_1d([0,1,2,3,4,5,6], 4), [4,5,6,0,1,2,3])
        npt.assert_array_equal(profiler.unreshuffle_1d([4,5,6,0,1,2,3], 4), [0,1,2,3,4,5,6])
        rs, i0 = profiler.reshuffle_2d([[0,1,2],[3,4,5]], (1,2))
        npt.assert_array_equal(rs, [5,4,3,0,1,2])
        npt.assert_array_equal(profiler.unreshuffle_2d([5,4,3,0,1,2], i0, (2,3)), [[0,1,2],[3,4,5]])
        rs, i0 = profiler.reshuffle_2d([[0,1,2],[3,4,5]], (0,1))
        npt.assert_array_equal(rs, [1,2,5,4,3,0])
        npt.assert_array_equal(profiler.unreshuffle_2d([1,2,5,4,3,0], i0, (2,3)), [[0,1,2],[3,4,5]])

    def test_profiler(self):
        # defining some dummy parameters and observables
        Parameter('tmp a');
        Parameter('tmp b');
        Parameter('tmp c');
        Parameter('tmp d');
        p = ParameterConstraints()
        p.set_constraint('tmp b', '2+-0.3')
        p.set_constraint('tmp c', '0.2+-0.1')
        p.set_constraint('tmp d', '1+-0.5')
        def prediction(wc_obj, par):
            return par['tmp a']**2+par['tmp b']+par['tmp c']+par['tmp d']**2
        flavio.Observable('tmp obs');
        Prediction('tmp obs', prediction);
        m=Measurement('tmp measurement')
        m.add_constraint(['tmp obs'],
                    flavio.statistics.probability.NormalDistribution(1, 0.2))
        # test 1D profiler
        fit_1d = FrequentistFit('test profiler 1d',
                                    p, ['tmp a'], ['tmp b', 'tmp c', 'tmp d'], ['tmp obs'])
        profiler_1d = profiler.Profiler1D(fit_1d, -10, 10)
        x, z, n = profiler_1d.run(steps=4)
        self.assertEqual(x.shape, (4,))
        self.assertEqual(z.shape, (4,))
        self.assertEqual(n.shape, (3, 4))
        npt.assert_array_equal(x, profiler_1d.x)
        npt.assert_array_equal(z, profiler_1d.log_profile_likelihood)
        npt.assert_array_equal(n, profiler_1d.profile_nuisance)
        pdat = profiler_1d.pvalue_prob_plotdata()
        npt.assert_array_equal(pdat['x'], x)
        # test multiprocessing
        for threads in [2, 3, 4]:
            xt, zt, nt = profiler_1d.run(steps=4, threads=threads)
            npt.assert_array_almost_equal(x, xt, decimal=4)
            npt.assert_array_almost_equal(z, zt, decimal=4)
            npt.assert_array_almost_equal(n, nt, decimal=4)
        with self.assertRaises(ValueError):
            profiler_1d.run(steps=4, threads=5)
        # test 2D profiler
        p.remove_constraint('d')
        fit_2d = FrequentistFit('test profiler 2d',
                                    p, ['tmp a', 'tmp d'], ['tmp b', 'tmp c'], ['tmp obs'])
        profiler_2d = profiler.Profiler2D(fit_2d, -10, 10, -10, 10)
        x, y, z, n = profiler_2d.run(steps=(3,4))
        self.assertEqual(x.shape, (3,))
        self.assertEqual(y.shape, (4,))
        self.assertEqual(z.shape, (3, 4))
        self.assertEqual(n.shape, (2, 3, 4))
        npt.assert_array_equal(x, profiler_2d.x)
        npt.assert_array_equal(y, profiler_2d.y)
        npt.assert_array_equal(z, profiler_2d.log_profile_likelihood)
        npt.assert_array_equal(n, profiler_2d.profile_nuisance)
        pdat = profiler_2d.contour_plotdata()
        npt.assert_array_almost_equal(pdat['z'], -2*(z-np.max(z)))
        # test multiprocessing
        for threads in [2, 5, 12]:
            xt, yt, zt, nt = profiler_2d.run(steps=(3,4))
            npt.assert_array_almost_equal(x, xt, decimal=4)
            npt.assert_array_almost_equal(y, yt, decimal=4)
            npt.assert_array_almost_equal(z, zt, decimal=4)
            npt.assert_array_almost_equal(n, nt, decimal=4)
        with self.assertRaises(ValueError):
            profiler_2d.run(steps=(3,4), threads=13)
        # delete dummy instances
        for p in ['tmp a', 'tmp b', 'tmp c', 'tmp d']:
            Parameter.del_instance(p)
        FrequentistFit.del_instance('test profiler 1d')
        Observable.del_instance('tmp obs')
        Measurement.del_instance('tmp measurement')
