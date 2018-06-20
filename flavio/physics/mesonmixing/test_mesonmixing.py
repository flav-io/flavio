import unittest
import numpy as np
from . import amplitude, observables
from math import sin, asin, cos, pi
from flavio.physics.eft import WilsonCoefficients
from flavio import Observable
from flavio.parameters import default_parameters
import copy
import flavio
import cmath
from wilson import Wilson

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
        -0.00686759,  0.        ,  0.        ,  0.        ,  0.53745488]])
# fixing a sign error in the above matrix
U_mb[4, 8] = -U_mb[4, 8]
U_mb[8, 4] = -U_mb[8, 4]
U_mb[5, 9] = -U_mb[5, 9]
U_mb[9, 5] = -U_mb[9, 5]
# reshaping
U_mb = U_mb.reshape((10,10))[[0,4,8,1,5,9,2,6],:][:,[0,4,8,1,5,9,2,6]]

class TestMesonMixing(unittest.TestCase):
    def test_bmixing(self):
        # just some trivial tests to see if calling the functions raises an error
        m12d = amplitude.M12(par, wc_B0, 'B0')
        m12s = amplitude.M12(par, wc_Bs, 'Bs')
        # check whether order of magnitudes of SM predictions are right
        ps = 1e-12*s
        self.assertAlmostEqual(observables.DeltaM_positive(wc_obj, par, 'B0')*ps, 0.55, places=0)
        self.assertAlmostEqual(observables.DeltaM_positive(wc_obj, par, 'Bs')*ps, 18, places=-1)
        self.assertAlmostEqual(observables.DeltaGamma_B(wc_obj, par, 'B0')/0.00261*ps, 1, places=0)
        self.assertAlmostEqual(observables.DeltaGamma_B(wc_obj, par, 'Bs')/0.088*ps, 1, places=0)
        self.assertAlmostEqual(observables.a_fs(wc_obj, par, 'B0')/-4.7e-4, 1, places=0)
        self.assertAlmostEqual(observables.a_fs(wc_obj, par, 'Bs')/2.22e-5, 1, places=0)
        self.assertAlmostEqual(observables.S_BJpsiK(wc_obj, par), 0.73, places=1)
        self.assertAlmostEqual(observables.S_Bspsiphi(wc_obj, par), asin(+0.038), places=2)
        # test classic formula: numerics of Wolfi's thesis
        w_par = par.copy()
        GF = w_par['GF']
        mW = w_par['m_W']
        mBs = w_par['m_Bs']
        fBs = 0.245
        w_par['f_Bs'] = fBs
        S0 = 2.31
        V = flavio.physics.ckm.ckm_wolfenstein(0.2254, 0.808, 0.177, 0.360)
        w_par['Vub'] = abs(V[0,2])
        w_par['Vcb'] = abs(V[1,2])
        w_par['Vus'] = abs(V[0,1])
        w_par['gamma'] = -cmath.phase(V[0,2])
        etaB = 0.55
        BBsh = 1.22 # 0.952 * 1.517
        w_par['bag_Bs_1'] = BBsh/1.5173
        M12 = (GF**2 * mW**2/12/pi**2 * etaB * mBs * fBs**2 * BBsh
               * S0 * (V[2,2] * V[2,1].conj())**2)
        self.assertAlmostEqual(amplitude.M12(w_par, wc_Bs, 'Bs')/M12, 1, delta=0.01)
        self.assertAlmostEqual(observables.DeltaM_positive(wc_obj, w_par, 'Bs')*ps, 18.3, delta=0.2)

    def test_dmixing(self):
        par_D = par.copy()
        wc_obj = flavio.WilsonCoefficients()
        # check correct limiting cases for vanishing mixing
        par_D['M12_D a_bb'] = 0
        par_D['M12_D a_bs'] = 0
        par_D['M12_D a_ss'] = 0
        par_D['Gamma12_D a_bb'] = 0
        par_D['Gamma12_D a_bs'] = 0
        par_D['Gamma12_D a_ss'] = 0
        self.assertEqual(flavio.Observable['x_D'].prediction_par(par_D, wc_obj), 0)
        self.assertEqual(flavio.Observable['y_D'].prediction_par(par_D, wc_obj), 0)
        self.assertEqual(flavio.Observable['phi_D'].prediction_par(par_D, wc_obj), 0)
        self.assertEqual(flavio.Observable['q/p_D'].prediction_par(par_D, wc_obj), 1)

    def test_dmixing(self):
        par_D = par.copy()
        wc_obj = flavio.WilsonCoefficients()
        par_D['M12_D a_bb'] = 0
        par_D['M12_D a_bs'] = 10
        par_D['M12_D a_ss'] = -0.1
        par_D['Gamma12_D a_bb'] = 0
        par_D['Gamma12_D a_bs'] = 10
        par_D['Gamma12_D a_ss'] = -0.3
        x12 = flavio.Observable['x12_D'].prediction_par(par_D, wc_obj)
        x12Im = flavio.Observable['x12Im_D'].prediction_par(par_D, wc_obj)
        phi12 = flavio.Observable['phi12_D'].prediction_par(par_D, wc_obj)
        self.assertAlmostEqual(x12 * sin(phi12), x12Im)

    def test_bmixing_classes(self):
        ps = 1e-12*s
        self.assertAlmostEqual(Observable['DeltaM_d'].prediction_central(c, wc_obj)*ps, 0.53, places=0)
        self.assertAlmostEqual(Observable['DeltaM_s'].prediction_central(c, wc_obj)*ps, 18, places=-1)
        self.assertAlmostEqual(Observable['DeltaGamma_d'].prediction_central(c, wc_obj)/0.00261*ps, 1, places=-1)
        self.assertAlmostEqual(Observable['DeltaGamma_s'].prediction_central(c, wc_obj)/0.088*ps, 1, places=-1)
        self.assertAlmostEqual(Observable['a_fs_d'].prediction_central(c, wc_obj)/-4.7e-4, 1, places=-1)
        self.assertAlmostEqual(Observable['a_fs_s'].prediction_central(c, wc_obj)/2.22e-5, 1, places=-1)
        self.assertAlmostEqual(Observable['S_psiK'].prediction_central(c, wc_obj), 0.73, places=-1)
        self.assertAlmostEqual(Observable['S_psiphi'].prediction_central(c, wc_obj), asin(+0.038), places=-1)

    def test_running(self):
        c_in = np.array([ 0.20910694,  0.77740198,  0.54696337,  0.46407456,  0.42482153,
        0.95717777,  0.62733321,  0.87053086])
        wc = flavio.WilsonCoefficients()
        wc_names = ['C{}_bsbs'.format(i)
                    for i in ['VLL', 'SLL', 'TLL', 'VRR', 'SRR', 'TRR', 'VLR', 'SLR']]
        wc_dict = dict(zip(wc_names, c_in))
        wc.set_initial(wc_dict, 173.3)
        c_out_dict = wc.get_wc('sbsb', 4.2, par)
        c_out = np.array([c_out_dict[k] for k in wc_names])
        c_out_U = np.dot(U_mb, c_in)
        for i, r in enumerate(c_out/c_out_U):
            if 'S' in wc_names[i] or 'T' in wc_names[i]:
                self.assertAlmostEqual(r, 1, delta=0.2,
                                       msg="Failed for {}".format(wc_names[i]))
            else:  # more precise
                self.assertAlmostEqual(r, 1, delta=0.1,
                                       msg="Failed for {}".format(wc_names[i]))
        # compare eta at 2 GeV to the values in table 2 of hep-ph/0102316
        par_bju = par.copy()
        par_bju['alpha_s'] = 0.118
        par_bju['m_b'] = 4.4
        c_out_bju_dict = wc.get_wc('sbsb', 2, par_bju, nf_out=5)
        c_out_bju = np.array([c_out_bju_dict[k] for k in wc_names])
        self.assertAlmostEqual(c_out_bju[0]/c_in[0], 0.788, delta=0.02)

    def test_common(self):
        # random values
        M12 = 0.12716600+0.08765385j
        G12 = -0.34399429-0.1490931j
        aM12 = abs(M12)
        aG12 = abs(G12)
        phi12 = cmath.phase(-M12/G12)
        qp = flavio.physics.mesonmixing.common.q_over_p(M12, G12)
        DM = flavio.physics.mesonmixing.common.DeltaM(M12, G12)
        DG = flavio.physics.mesonmixing.common.DeltaGamma(M12, G12)
        # arXiv:0904.1869
        self.assertAlmostEqual(DM**2-1/4.*DG**2, 4*aM12**2-aG12**2, places=10) # (35)
        self.assertAlmostEqual(qp, -(DM+1j*DG/2)/(2*M12-1j*G12), places=10) # (37)
        self.assertAlmostEqual(qp, -(2*M12.conjugate()-1j*G12.conjugate())/(DM+1j*DG/2), places=10) # (37)
        self.assertAlmostEqual(DM*DG, -4*(M12*G12.conjugate()).real, places=10) # (36)
        self.assertAlmostEqual(DM*DG, 4*aM12*aG12*cos(phi12), places=10) # (39)

    def test_np(self):
        # Bd
        CVSM = flavio.physics.mesonmixing.wilsoncoefficient.cvll_d(par, 'B0', scale=80)[0]
        w = Wilson({'CVRR_bdbd': CVSM}, 80, 'WET', 'flavio')
        self.assertAlmostEqual(
            flavio.np_prediction('DeltaM_d', w) / flavio.sm_prediction('DeltaM_d'),
            2,
            delta=0.15)  # difference due to NNLO evolution of SM contribution
        # Bs
        CVSM = flavio.physics.mesonmixing.wilsoncoefficient.cvll_d(par, 'Bs', scale=80)[0]
        w = Wilson({'CVRR_bsbs': CVSM}, 80, 'WET', 'flavio')
        self.assertAlmostEqual(
            flavio.np_prediction('DeltaM_s', w) / flavio.sm_prediction('DeltaM_s'),
            2,
            delta=0.15)  # difference due to NNLO evolution of SM contribution
        # K0
        CVSM = flavio.physics.mesonmixing.wilsoncoefficient.cvll_d(par, 'K0', scale=80)[0]
        w = Wilson({'CVRR_sdsd': CVSM}, 80, 'WET', 'flavio')
        self.assertAlmostEqual(
            flavio.np_prediction('eps_K', w) / flavio.sm_prediction('eps_K'),
            2,
            delta=0.15)  # difference due to NNLO evolution of SM contribution + charm contribution
