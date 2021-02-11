import unittest
import flavio

class TestCitations(unittest.TestCase):
    def test_citations(self):
        citations = flavio.default_citations
        # Default paper should be in the papers to cite
        self.assertIn("Straub:2018kue", citations.toset)
        citations._reset()
        self.assertNotIn("Beneke:2003az", citations.toset)
        flavio.sm_prediction("DeltaGamma_s")
        self.assertIn("Beneke:2003az", citations.toset)

    def test_register(self):
        citations = flavio.default_citations
        citations.register("fakename:2020abc")
        self.assertIn("fakename:2020abc", citations.toset)

    def test_theory_citations(self):
        DGs_citations = flavio.Observable["DeltaGamma_s"].theory_citations()
        self.assertNotIn("Straub:2018kue", DGs_citations)
        self.assertIn("Beneke:2003az", DGs_citations)
