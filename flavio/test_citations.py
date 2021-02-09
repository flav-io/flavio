import unittest
import flavio

class TestCitations(unittest.TestCase):
    def test_citations(self):
        citations = flavio.default_citations
        # Default paper should be in the papers to cite
        self.assertIn("Straub:2018kue", citations._papers_to_cite)
        citations._reset()
        self.assertNotIn("Beneke:2003az", citations._papers_to_cite)
        flavio.sm_prediction("DeltaGamma_s")
        self.assertIn("Beneke:2003az", citations._papers_to_cite)

    def test_register(self):
        citations = flavio.default_citations
        citations.register("fakename:2020abc")
        self.assertIn("fakename:2020abc", citations._papers_to_cite)

    def test_print_citations(self):
        citations = flavio.default_citations
        citations._reset()
        citations.register("fakename:2020abc")
        citations.register("fakename:2018def")
        self.assertSetEqual(set(["fakename:2020abc", "fakename:2018def"]), set(flavio.print_citations().split(",")))
        flavio.default_citations._reset()
        self.assertEqual("", flavio.print_citations())

    def test_theory_citations(self):
        DGs_citations = flavio.Observable["DeltaGamma_s"].theory_citations().split(",")
        self.assertNotIn("Straub:2018kue", DGs_citations)
        self.assertIn("Beneke:2003az", DGs_citations)
