import unittest
import flavio
from flavio.physics.dileptons import pplnu
from wilson import wcxf
from flavio.physics.dileptons.test_ppll import uses_pdf
import numpy as np
from flavio.config import config

par = flavio.default_parameters.get_central_all()
par2 = par.copy()
par2['s2w'] = 0.22137905667111532  # correction factor to match the madgraph input scheme
GeVtopb = 0.3894*10**9

class Test_QQLNU(unittest.TestCase):
    @uses_pdf
    def test_sm_spectrum(self):
        r"""Compare the SM $m_T$ spectrum to madgraph events in 10 bins from 200 GeV to 2 TeV
            MG events simulated with NNPDF30_nnlo_as_0118
            2% acceptable error"""

        spectrum_mg = np.asarray([3.41707366e-02, 1.02592426e-02, 3.20641729e-03, 9.63481603e-04,
                                    2.81233386e-04, 8.12019322e-05, 2.13711295e-05, 5.30226309e-06,
                                    1.14687576e-06])
        # Number of SM events generated in MG [66095., 25637., 33458., 48654., 18351.,  6849., 59869., 32043., 9044.]

        s = 13e3**2
        logbins = np.linspace(np.log10(200),np.log10(2000),10)
        bins = 10**logbins
        nbins = len(bins)-1
        for i in range(nbins):
            center = 0.5*(bins[i]+bins[i+1])
            width = bins[i+1]-bins[i]
            spectrum = pplnu.sigma_qqlnu_int(s, bins[i], bins[i+1], 'mu', 0, par2, center**2, 0, newphys=False)*GeVtopb/width
            err = (spectrum-spectrum_mg[i])/spectrum_mg[i]
            self.assertAlmostEqual(err,0,delta=0.02,msg=f'error in bin {i}: {err}')

    @uses_pdf
    def test_qqlnu_sm(self):
        """Test the SM pediction for the R ratio from 1 to 2 TeV"""
        R = flavio.sm_prediction('R_13(pp->munu)', 1e3, 2e3)
        self.assertEqual(R, 1,msg=f'SM prediction for R ratio: {R} (has to be 1)')

    @uses_pdf
    def test_qqlnu_np_sm(self):
        r"""Test the R ratio with presence of wilson coefficients set to zero"""
        wc_obj = flavio.WilsonCoefficients()
        R = pplnu.R_sigma_qqlnu_int(13e3**2, 1300, 1800, 'mu', wc_obj, par)
        self.assertEqual(R, 1,msg=f'SM prediction for R ratio: {R} (has to be 1)')

    @uses_pdf
    def test_qqlnu_np_vector(self):
        r"""Test the R ratio for one specific vector WC
            comparing to results from MadGraph using NNPDF30_nnlo_as_0118 and SMEFTsim_general_MwScheme_UFO model
            2% acceptable error"""

        R_mg = np.asarray([ 8.76973438, 12.82963675, 18.01933393])
        # Number of NP events generated in MG [25183, 16648, 11444]
        # Number of SM events generated in MG [23536, 10207, 4851]

        bins = np.asarray([1200., 1400., 1600., 1800.])
        nbins = len(bins)-1

        for i in range(nbins):
            center = 0.5*(bins[i]+bins[i+1])
            wc = wcxf.WC('SMEFT', 'Warsaw up', center, {'lq3_2211': 1e-7})
            wc_obj = flavio.WilsonCoefficients()
            wc_obj.set_initial_wcxf(wc)
            R = pplnu.R_sigma_qqlnu_int(13e3**2, bins[i], bins[i+1], 'mu', wc_obj, par2)
            err = (R-R_mg[i])/R_mg[i]
            self.assertAlmostEqual(err,0,delta=0.02,msg=f'error in bin {i}: {err}')

    @uses_pdf
    def test_qqlnu_np_scalar(self):
        r"""Test the R ratio for one specific scalar WC
            comparing to results from MadGraph using NNPDF30_nnlo_as_0118 and SMEFTsim_general_MwScheme_UFO model
            2% acceptable error"""

        R_mg = np.asarray([1.00938347, 1.01333147, 1.01762706])
        # Number of NP events generated in MG [25603, 15708, 9833]
        # Number of SM events generated in MG [23536, 10207, 4851]

        bins = np.asarray([1200., 1400., 1600., 1800.])
        nbins = len(bins)-1

        for i in range(nbins):
            center = 0.5*(bins[i]+bins[i+1])
            wc = wcxf.WC('SMEFT', 'Warsaw up', center, {'lequ1_2232': 1e-7})
            wc_obj = flavio.WilsonCoefficients()
            wc_obj.set_initial_wcxf(wc)
            R = pplnu.R_sigma_qqlnu_int(13e3**2, bins[i], bins[i+1], 'mu', wc_obj, par2)
            err = (R-R_mg[i])/R_mg[i]
            self.assertAlmostEqual(err,0,delta=0.02,msg=f'error in bin {i}: {err}')

    @uses_pdf
    def test_qqlnu_np_tensor(self):
        r"""Test the R ratio for one specific tensor WC
            comparing to results from MadGraph using NNPDF30_nnlo_as_0118 and SMEFTsim_general_MwScheme_UFO model
            2% acceptable error"""

        R_mg = np.asarray([1.42841172, 1.63238727, 1.87684397])
        # Number of NP events generated in MG [25779, 16499, 10905]
        # Number of SM events generated in MG [23536, 10207, 4851]

        bins = np.asarray([1200., 1400., 1600., 1800.])
        nbins = len(bins)-1

        for i in range(nbins):
            center = 0.5*(bins[i]+bins[i+1])
            wc = wcxf.WC('SMEFT', 'Warsaw up', center, {'lequ3_2212': 1e-7})
            wc_obj = flavio.WilsonCoefficients()
            wc_obj.set_initial_wcxf(wc)
            R = pplnu.R_sigma_qqlnu_int(13e3**2, bins[i], bins[i+1], 'mu', wc_obj, par2)
            err = (R-R_mg[i])/R_mg[i]
            self.assertAlmostEqual(err,0,delta=0.02,msg=f'error in bin {i}: {err}')
