import unittest
import flavio
from flavio.physics.dileptons import ppll, partondist
from wilson import wcxf
import numpy as np
from flavio.config import config

par = flavio.default_parameters.get_central_all()
par2 = par.copy()
par2['s2w'] = 0.221841          # correction factor to match the madgraph input scheme
par2['alpha_e'] = 0.00772585    # correction factor to match the madgraph input scheme
GeVtopb = 0.3894*10**9

pdf_set_for_tests = 'NNPDF30_nnlo_as_0118'
config['PDF set']['dileptons']['name'] = pdf_set_for_tests

def uses_pdf(func):
    def inner(*args, **kwargs):
        partondist.get_pdf(pdf_set_for_tests, download=True)
        return func(*args, **kwargs)
    return inner

class Test_QQLL(unittest.TestCase):
    @uses_pdf
    def test_sm_spectrum(self):
        r"""Compare the SM $m_{\ell\ell}$ spectrum from flavio to madgraph prediction in 10 bins from 200 GeV to 2 TeV
            MG events simulated with NNPDF30_nnlo_as_0118 PDFs
            2% acceptable error"""

        spectrum_mg = np.asarray([2.66250518e-02, 8.76624628e-03, 2.92981721e-03, 9.64842852e-04, 3.12854454e-04, 9.77018348e-05, 2.84022912e-05, 8.08026977e-06, 2.02974822e-06])
        # Number of SM events generated in MG [60169., 25284., 10791., 25514., 52059., 20750., 42577., 45628., 14659.]

        s = 13e3**2
        logbins = np.linspace(np.log10(200),np.log10(1800),10)
        bins = 10**logbins
        nbins = len(bins)-1

        for i in range(nbins):
            center = 0.5*(bins[i]+bins[i+1])
            width = bins[i+1]-bins[i]
            spectrum = ppll.sigma_qqll_int(s, bins[i], bins[i+1], 'mu', center**2, 0, par2, 0, newphys=False)*GeVtopb/width
            err = (spectrum-spectrum_mg[i])/spectrum_mg[i]
            self.assertAlmostEqual(err,0,delta=0.02,msg=f'error in bin {i}: {err}')

    @uses_pdf
    def test_qqll_sm(self):
        """Test the SM pediction for the R ratio from 1 to 2 TeV"""

        R = flavio.sm_prediction('R_13(pp->mumu)', 1e3, 2e3)
        self.assertEqual(R, 1,msg=f'SM prediction for R ratio: {R} (has to be 1)')

    @uses_pdf
    def test_qqll_np_sm(self):
        r"""Test the R ratio with presence of wilson coefficients set to zero"""

        wc_obj = flavio.WilsonCoefficients()
        R = ppll.R_sigma_qqll_int(13e3**2, 1300, 1800, 'mu', wc_obj, par)
        self.assertEqual(R, 1,msg=f'SM prediction for R ratio: {R} (has to be 1)')

    @uses_pdf
    def test_qqll_np_vector(self):
        r"""Testthe R ratio for one specific vector WC
            comparing to results from MadGraph using NNPDF30_nnlo_as_0118, SMEFTsim_general_alphaScheme_UFO
            2% acceptable error"""

        R_mg = np.asarray([1.1998640111246237, 1.6266639004553087, 2.4701438261611193])
        # Number of NP events generated in MG [37802, 24730, 16780]
        # Number of SM events generated in MG [58147, 23848, 10473]

        bins = np.asarray([1000., 1200., 1400., 1600.])
        nbins = len(bins)-1

        for i in range(nbins):
            center = 0.5*(bins[i]+bins[i+1])
            wc = wcxf.WC('SMEFT', 'Warsaw up', center, {'lq1_2211': 1e-7})
            wc_obj = flavio.WilsonCoefficients()
            wc_obj.set_initial_wcxf(wc)
            R = ppll.R_sigma_qqll_int(13e3**2, bins[i], bins[i+1], 'mu', wc_obj, par2)
            err = (R-R_mg[i])/R_mg[i]
            self.assertAlmostEqual(err,0,delta=0.02,msg=f'error in bin {i}: {err}')

    @uses_pdf
    def test_qqll_np_scalar(self):
        r"""Test the R ratio for one specific scalar WC
            comparing to results from MadGraph using NNPDF30_nnlo_as_0118, SMEFTsim_general_alphaScheme_UFO
            2% acceptable error"""

        R_mg = np.asarray([1.0159054527417566, 1.0257049317291311, 1.0386032816907145])
        # Number of NP events generated in MG [38339, 25414, 16752]
        # Number of SM events generated in MG [58147, 23848, 10473]

        bins = np.asarray([1000., 1200., 1400., 1600.])
        nbins = len(bins)-1

        for i in range(nbins):
            center = 0.5*(bins[i]+bins[i+1])
            wc = wcxf.WC('SMEFT', 'Warsaw up', center, {'ledq_2223': 1e-7})
            wc_obj = flavio.WilsonCoefficients()
            wc_obj.set_initial_wcxf(wc)
            R = ppll.R_sigma_qqll_int(13e3**2, bins[i], bins[i+1], 'mu', wc_obj, par2)
            err = (R-R_mg[i])/R_mg[i]
            self.assertAlmostEqual(err,0,delta=0.02,msg=f'error in bin {i}: {err}')

    @uses_pdf
    def test_qqll_np_tensor(self):
        r"""Test the R ratio for one specific tensor WC
            comparing to results from MadGraph using NNPDF30_nnlo_as_0118, SMEFTsim_general_alphaScheme_UFO
            2% acceptable error"""

        R_mg = np.asarray([2.1352791354283926, 3.0789137670371165, 4.609226551698083])
        # Number of NP events generated in MG [32345, 24282, 18505]
        # Number of SM events generated in MG [58147, 23848, 10473]

        bins = np.asarray([1000., 1200., 1400., 1600.])
        nbins = len(bins)-1

        for i in range(nbins):
            center = 0.5*(bins[i]+bins[i+1])
            wc = wcxf.WC('SMEFT', 'Warsaw up', center, {'lequ3_2212': 1e-7})
            wc_obj = flavio.WilsonCoefficients()
            wc_obj.set_initial_wcxf(wc)
            R = ppll.R_sigma_qqll_int(13e3**2, bins[i], bins[i+1], 'mu', wc_obj, par2)
            err = (R-R_mg[i])/R_mg[i]
            self.assertAlmostEqual(err,0,delta=0.02,msg=f'error in bin {i}: {err}')
