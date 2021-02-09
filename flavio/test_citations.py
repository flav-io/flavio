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

    def test_tex_citation_string(self):
        citations = flavio.default_citations
        citations._reset()
        citations.register("fakename:2020abc")
        citations.register("fakename:2018def")
        # I think either order is possible, since Python sets have no guaranteed ordering
        self.assertTrue(   r"\cite{fakename:2020abc,fakename:2018def}" == citations.tex_citation_string()
                        or r"\cite{fakename:2018def,fakename:2020abc}" == citations.tex_citation_string())
        flavio.default_citations._reset()
        self.assertEqual(r"\cite{}", flavio.default_citations.tex_citation_string())

    def test_theory_citations(self):
        DGs_citations = flavio.Observable["DeltaGamma_s"].theory_citations()
        self.assertNotIn("Straub:2018kue", DGs_citations)
        self.assertIn("Beneke:2003az", DGs_citations)
