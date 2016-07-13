import unittest
import numpy as np
from . import amplitude, rge, observables
from math import sin, asin
from flavio.physics.eft import WilsonCoefficients
from flavio import Observable
from flavio.parameters import default_parameters
import copy
import flavio

s = 1.519267515435317e+24

c = copy.deepcopy(default_parameters)
par = c.get_central_all()

wc_obj = WilsonCoefficients()
wc_B0 = wc_obj.get_wc('bdbd', 4.2, par)
wc_Bs = wc_obj.get_wc('bsbs', 4.2, par)
wc_K = wc_obj.get_wc('sdsd', 2, par)

# this is the DeltaF=2 evolution matrix from mt to 4.2 GeV as obtained
# from the formulae in hep-ph/0102316
U_mb = np.array([[ 0.83693251,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.83693251,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.91882327,  0.        ,  0.        ,
         0.        , -0.04335413,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.91882327,  0.        ,
         0.        ,  0.        , -0.04335413,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  1.68277548,
         0.        ,  0.        ,  0.        ,  2.06605957,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         1.68277548,  0.        ,  0.        ,  0.        ,  2.06605957],
       [ 0.        ,  0.        , -0.92932569,  0.        ,  0.        ,
         0.        ,  2.31878868,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        , -0.92932569,  0.        ,
         0.        ,  0.        ,  2.31878868,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        , -0.00686759,
         0.        ,  0.        ,  0.        ,  0.53745488,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        -0.00686759,  0.        ,  0.        ,  0.        ,  0.53745488]]).reshape((10,10))[[0,4,8,1,5,9,2,6],:][:,[0,4,8,1,5,9,2,6]]

class TestMesonMixing(unittest.TestCase):
    def test_bmixing(self):
        # just some trivial tests to see if calling the functions raises an error
        m12d = amplitude.M12_d(par, wc_B0, 'B0')
        m12s = amplitude.M12_d(par, wc_Bs, 'Bs')
        # check whether order of magnitudes of SM predictions are right
        ps = 1e-12*s
        self.assertAlmostEqual(observables.DeltaM(wc_obj, par, 'B0')*ps, 0.55, places=0)
        self.assertAlmostEqual(observables.DeltaM(wc_obj, par, 'Bs')*ps, 18, places=-1)
        self.assertAlmostEqual(observables.DeltaGamma(wc_obj, par, 'B0')/0.00261*ps, 1, places=0)
        self.assertAlmostEqual(observables.DeltaGamma(wc_obj, par, 'Bs')/0.088*ps, 1, places=0)
        self.assertAlmostEqual(observables.a_fs(wc_obj, par, 'B0')/-4.7e-4, 1, places=0)
        self.assertAlmostEqual(observables.a_fs(wc_obj, par, 'Bs')/2.22e-5, 1, places=0)
        self.assertAlmostEqual(observables.S_BJpsiK(wc_obj, par), 0.73, places=1)
        self.assertAlmostEqual(observables.S_Bspsiphi(wc_obj, par), asin(+0.038), places=2)

    def test_bmixing_classes(self):
        ps = 1e-12*s
        self.assertAlmostEqual(Observable.get_instance('DeltaM_d').prediction_central(c, wc_obj)*ps, 0.53, places=0)
        self.assertAlmostEqual(Observable.get_instance('DeltaM_s').prediction_central(c, wc_obj)*ps, 18, places=-1)
        self.assertAlmostEqual(Observable.get_instance('DeltaGamma_d').prediction_central(c, wc_obj)/0.00261*ps, 1, places=-1)
        self.assertAlmostEqual(Observable.get_instance('DeltaGamma_s').prediction_central(c, wc_obj)/0.088*ps, 1, places=-1)
        self.assertAlmostEqual(Observable.get_instance('a_fs_d').prediction_central(c, wc_obj)/-4.7e-4, 1, places=-1)
        self.assertAlmostEqual(Observable.get_instance('a_fs_s').prediction_central(c, wc_obj)/2.22e-5, 1, places=-1)
        self.assertAlmostEqual(Observable.get_instance('S_psiK').prediction_central(c, wc_obj), 0.73, places=-1)
        self.assertAlmostEqual(Observable.get_instance('S_psiphi').prediction_central(c, wc_obj), asin(+0.038), places=-1)

    def test_running(self):
        c_in = np.array([ 0.20910694,  0.77740198,  0.54696337,  0.46407456,  0.42482153,
        0.95717777,  0.62733321,  0.87053086])
        c_out = flavio.physics.running.running.get_wilson(par, c_in, wc_obj.rge_derivative['bsbs'], 173.3, 4.2)
        c_out_U = np.dot(U_mb, c_in)
        np.testing.assert_almost_equal(c_out/c_out_U, np.ones(8), decimal=2)
        # compare eta at 2 GeV to the values in table 2 of hep-ph/0102316
        par_bju = par.copy()
        par_bju['alpha_s'] = 0.118
        par_bju['m_b'] = 4.4
        c_out_bju = rge.run_wc_df2(par_bju, c_in, 166., 2)
        self.assertAlmostEqual(c_out_bju[0]/c_in[0], 0.788, places=2)
