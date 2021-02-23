import unittest
import yaml
import pkgutil
import flavio
from flavio.util import extract_citations

class TestCitations(unittest.TestCase):
    def test_citations(self):
        citations = flavio.citations
        # Default paper should be in the papers to cite
        self.assertIn("Straub:2018kue", citations.set)
        flavio.sm_prediction("DeltaGamma_s")
        self.assertIn("Beneke:2003az", citations.set)
        citations.register("fakename:2020abc")
        self.assertIn("fakename:2020abc", citations.set)
        citations.reset()
        self.assertNotIn("Beneke:2003az", citations.set)
        self.assertNotIn("fakename:2020abc", citations.set)
        self.assertIn("Straub:2018kue", citations.set)

    def test_clear_citations(self):
        citations = flavio.citations
        citations.register("fakename:2020abc")
        citations.clear()
        self.assertNotIn("fakename:2020abc", citations.set)
        self.assertNotIn("Straub:2018kue", citations.set)

    def test_theory_citations(self):
        DGs_citations = flavio.Observable["DeltaGamma_s"].theory_citations()
        self.assertNotIn("Straub:2018kue", DGs_citations)
        self.assertIn("Beneke:2003az", DGs_citations)

    def test_multithread(self):
        obs = ("DeltaGamma_s", "m_W")
        flavio.citations.reset()
        flavio.sm_covariance(obs, threads=1)
        cites_singlethread = flavio.citations.set
        flavio.citations.reset()
        flavio.sm_covariance(obs, threads=4)
        cites_multithread = flavio.citations.set
        self.assertSetEqual(cites_singlethread, cites_multithread)

    def test_citation_yml_update(self):
        cites_found = extract_citations()
        cites_recorded = set(yaml.safe_load(
                pkgutil.get_data('flavio', 'data/citations.yml')
        ))
        self.assertSetEqual(cites_found, cites_recorded,
                            msg="Update citations.yml by running update_citations.py")
