import unittest
import flavio

class TestCitations(unittest.TestCase):
    def test_citations(self):
        citations = flavio.citations
        # Default paper should be in the papers to cite
        self.assertIn("Straub:2018kue", citations.set)
        flavio.sm_prediction("DeltaGamma_s")
        self.assertIn("Beneke:2003az", citations.set)
        citations.reset()
        self.assertNotIn("Beneke:2003az", citations.set)

    def test_register(self):
        citations = flavio.citations
        citations.register("fakename:2020abc")
        self.assertIn("fakename:2020abc", citations.set)

    def test_theory_citations(self):
        DGs_citations = flavio.Observable["DeltaGamma_s"].theory_citations()
        self.assertNotIn("Straub:2018kue", DGs_citations)
        self.assertIn("Beneke:2003az", DGs_citations)
