import unittest
import numpy as np
import flavio

class TestKaonFFs(unittest.TestCase):
    def test_kaon_ffs(self):
        # check if this raises an error
        par = flavio.default_parameters.get_central_all()
        flavio.physics.kdecays.formfactors.fp0_dispersive(q2=0.1, par=par)
        flavio.physics.kdecays.formfactors.fT_pole(q2=0.1, par=par)
        # check that f0 and f+ are equal at q2=0
        ff0 = flavio.physics.kdecays.formfactors.ff_dispersive_pole(wc_obj=False, q2=0., par_dict=par)
        self.assertEqual(ff0['f+'], ff0['f0'])
        # check that f+ at q2=0 equals the corresponding parameter
        self.assertEqual(ff0['f+'], par['K->pi f+(0)'])
