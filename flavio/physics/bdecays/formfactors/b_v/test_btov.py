import unittest
from flavio.physics.bdecays.formfactors.b_v import bsz_parameters, lattice_parameters, cln
from flavio.classes import Implementation
from flavio.parameters import default_parameters
import copy

# FF values based on central values from 1811.00983 as obtained with EOS
test_eos_ff = {'pi': {1.5: {'f0': 0.20617313988681782, 'f+': 0.21680149224287226}, 6: {'f0': 0.2438330298045479, 'f+': 0.3004819340685217}}, 'D*': {1.5: {'A12': 0.19752354730048488, 'T2': 0.6403093357932056, 'V': 0.7202354964684643, 'T23': 0.8294490273576515, 'T1': 0.6652318425233219, 'A0': 0.7230006484800593, 'A1': 0.6202446275095023}, 6: {'A12': 0.22933336108737656, 'T2': 0.6727838222595557, 'V': 0.8426391243133804, 'T23': 0.912429103591902, 'T1': 0.7903787568156171, 'A0': 0.9013317472728795, 'A1': 0.6935347915881062}}, 'K': {1.5: {'f0': 0.3430632833575107, 'f+': 0.35715663480062054}, 6: {'f0': 0.39283486258783235, 'f+': 0.4671080094503594}}, 'rho': {1.5: {'A12': 0.24959844928281663, 'T2': 0.24181804607576635, 'V': 0.29165579022587984, 'T23': 0.5748008550726219, 'T1': 0.25574046077962087, 'A0': 0.3173049328262423, 'A1': 0.21911594799415993}, 6: {'A12': 0.2690738575432775, 'T2': 0.25352450555973377, 'V': 0.36881156694745687, 'T23': 0.629355146736795, 'T1': 0.32499988698730087, 'A0': 0.4148275948811504, 'A1': 0.23397806916756847}}, 'K*': {1.5: {'A12': 0.2512344471458036, 'T2': 0.3460508135685656, 'V': 0.41395241458255816, 'T23': 0.6676699778453463, 'T1': 0.3651561001308803, 'A0': 0.3746153403408867, 'A1': 0.3100091228571904}, 6: {'A12': 0.27101140476850843, 'T2': 0.37179901507005286, 'V': 0.5437850389870088, 'T23': 0.7455209444479692, 'T1': 0.4710838481302146, 'A0': 0.5014073911808256, 'A1': 0.3422466383471718}}, 'D': {1.5: {'f0': 0.7022104214739393, 'f+': 0.7245750744029429}, 6: {'f0': 0.7772907542773136, 'f+': 0.883094727230034}}}

class TestBtoV(unittest.TestCase):

    def test_cln(self):
        c = copy.deepcopy(default_parameters)
        par = c.get_central_all()
        cln.ff('B->D*', 1, par, 4.8)
        Implementation['B->D* CLN'].get_central(constraints_obj=c, wc_obj=None, q2=1)
        ff0 = Implementation['B->D* CLN'].get_central(constraints_obj=c, wc_obj=None, q2=0)
        mB = par['m_B0']
        mV = par['m_D*+']
        # check that exact kinematic relation at q^2=0 is fulfilled
        self.assertAlmostEqual(ff0['A0'], 8*mB*mV/(mB**2-mV**2)*ff0['A12'], places=12)

    def test_bsz3(self):
        c = copy.deepcopy(default_parameters)
        bsz_parameters.bsz_load_v1_lcsr(c)
        # compare to numbers in table 4 of arXiv:1503.05534v1
        # B->K* all FFs
        ffbsz3 = Implementation['B->K* BSZ3'].get_central(constraints_obj=c, wc_obj=None, q2=0)
        self.assertAlmostEqual(ffbsz3['A0'], 0.391, places=2)
        self.assertAlmostEqual(ffbsz3['A1'], 0.289, places=3)
        self.assertAlmostEqual(ffbsz3['A12'], 0.281, places=1)
        self.assertAlmostEqual(ffbsz3['V'], 0.366, places=3)
        self.assertAlmostEqual(ffbsz3['T1'], 0.308, places=3)
        self.assertAlmostEqual(ffbsz3['T23'], 0.793, places=3)
        self.assertAlmostEqual(ffbsz3['T1'], ffbsz3['T2'], places=16)
        # A1 for the remaining transitions
        ffbsz3 = Implementation['B->rho BSZ3'].get_central(constraints_obj=c, wc_obj=None, q2=0)
        self.assertAlmostEqual(ffbsz3['A1'], 0.267, places=3)
        ffbsz3 = Implementation['B->omega BSZ3'].get_central(constraints_obj=c, wc_obj=None, q2=0)
        self.assertAlmostEqual(ffbsz3['A1'], 0.237, places=3)
        ffbsz3 = Implementation['Bs->phi BSZ3'].get_central(constraints_obj=c, wc_obj=None, q2=0)
        self.assertAlmostEqual(ffbsz3['A1'], 0.315, places=3)
        ffbsz3 = Implementation['Bs->K* BSZ3'].get_central(constraints_obj=c, wc_obj=None, q2=0)
        self.assertAlmostEqual(ffbsz3['A1'], 0.246, places=3)

    def test_gkvd(self):
        # compare to numbers of arXiv:1811.00983
        c = copy.deepcopy(default_parameters)
        for q2 in [1.5, 6]:
            for ff in ['A0', 'A1', 'A12', 'V', 'T1', 'T2', 'T23']:
                for V in ['K*', 'D*', 'rho']:
                    fit = 'LCSR' if V == 'rho' else 'LCSR-Lattice'
                    bsz_parameters.gkvd_load('v1', fit, ('B->{}'.format(V),), c)
                    ffbsz3 = Implementation['B->{} BSZ3'.format(V)].get_central(constraints_obj=c, wc_obj=None, q2=q2)
                    self.assertAlmostEqual(ffbsz3[ff] / test_eos_ff[V][q2][ff],
                                           1,
                                           places=2,
                                           msg="Failed for {} in B->{} at q2={}".format(ff, V, q2))

    def test_lattice(self):
        c = copy.deepcopy(default_parameters)
        lattice_parameters.lattice_load(c)
        fflatt = Implementation['B->K* SSE'].get_central(constraints_obj=c, wc_obj=None, q2=12.)
        self.assertAlmostEqual(fflatt['V'], 0.84, places=2)
        self.assertAlmostEqual(fflatt['A0'], 0.861, places=3)
        self.assertAlmostEqual(fflatt['A1'], 0.440, places=3)
        self.assertAlmostEqual(fflatt['A12'], 0.339, places=3)
        self.assertAlmostEqual(fflatt['T1'], 0.711, places=3)
        self.assertAlmostEqual(fflatt['T2'], 0.433, places=3)
        self.assertAlmostEqual(fflatt['T23'], 0.809, places=3)
        fflatt = Implementation['Bs->phi SSE'].get_central(constraints_obj=c, wc_obj=None, q2=12.)
        self.assertAlmostEqual(fflatt['V'], 0.767, places=2)
        self.assertAlmostEqual(fflatt['A0'], 0.907, places=2)
        self.assertAlmostEqual(fflatt['A1'], 0.439, places=2)
        self.assertAlmostEqual(fflatt['A12'], 0.321, places=2)
        self.assertAlmostEqual(fflatt['T1'], 0.680, places=2)
        self.assertAlmostEqual(fflatt['T2'], 0.439, places=2)
        self.assertAlmostEqual(fflatt['T23'], 0.810, places=2)
        fflatt = Implementation['Bs->K* SSE'].get_central(constraints_obj=c, wc_obj=None, q2=12.)
        self.assertAlmostEqual(fflatt['V'], 0.584, places=3)
        self.assertAlmostEqual(fflatt['A0'], 0.884, places=3)
        self.assertAlmostEqual(fflatt['A1'], 0.370, places=3)
        self.assertAlmostEqual(fflatt['A12'], 0.321, places=3)
        self.assertAlmostEqual(fflatt['T1'], 0.605, places=3)
        self.assertAlmostEqual(fflatt['T2'], 0.383, places=3)
        self.assertAlmostEqual(fflatt['T23'], 0.743, places=3)

class TestCLN2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.c = default_parameters.copy()
        cls.imp = Implementation['B->D* CLN']

    def test_q2_1(self):
        ff = self.__class__.imp.get_central(constraints_obj=self.__class__.c, wc_obj=None, q2=1)
        for k, v in ff.items():
            self.assertTrue(v > 0, msg="Failed for {}".format(k))

    def test_q2_0_A(self):
        ff = self.__class__.imp.get_central(constraints_obj=self.__class__.c, wc_obj=None, q2=0)
        for k, v in ff.items():
            self.assertTrue(v > 0, msg="Failed for {}".format(k))
        par = self.__class__.c.get_central_all()
        mB = par['m_B0']
        mV = par['m_D*+']
        # check that exact kinematic relation at q^2=0 is fulfilled
        self.assertAlmostEqual(ff['A0'], 8*mB*mV/(mB**2-mV**2)*ff['A12'],
                               places=10)

    def test_q2_0_T(self):
        ff = self.__class__.imp.get_central(constraints_obj=self.__class__.c, wc_obj=None, q2=0)
        par = self.__class__.c.get_central_all()
        mB = par['m_B0']
        mV = par['m_D*+']
        # check that exact kinematic relation at q^2=0 is fulfilled
        self.assertEqual(ff['T1'], ff['T2'])
        ff = self.__class__.imp.get_central(constraints_obj=self.__class__.c, wc_obj=None, q2=(mB-mV)**2)
