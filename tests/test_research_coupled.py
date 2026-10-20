import tempfile
import unittest
from pathlib import Path

from optima.rag.research.config import ConfigError, PipelineConfig, RepresentationConfig
from optima.rag.research.corpus import load_corpus
from optima.rag.research.dense import HashingEncoder
from optima.rag.research.pipeline import ResearchPipeline
from optima.rag.research.plan import PlanBuilder
from optima.rag.research.representations import build_texts, coupled_text

from research_fixtures import write_fixture


class CoupledTextTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        paths = write_fixture(Path(self._tmp.name))
        self.paths = paths
        self.corpus = load_corpus(paths["base"], paths["enriched"])
        self.cfg = RepresentationConfig()
        self.enriched = [r for r in self.corpus.records.values() if r.has_semantic]
        self.plain = [r for r in self.corpus.records.values() if not r.has_semantic]

    def test_chunk_contains_header_once_description_and_source(self):
        record = self.enriched[0]
        text = coupled_text(record, self.cfg)
        self.assertEqual(text.count("Function:"), 1)
        self.assertIn("Description:", text)
        self.assertIn("Source:", text)
        self.assertIn(record.source_code.strip()[:20], text)
        purpose = record.semantic.get("purpose")
        self.assertTrue(purpose and str(purpose)[:15] in text)

    def test_order_controls_which_part_comes_first(self):
        record = self.enriched[0]
        first = coupled_text(record, self.cfg)
        other = coupled_text(record, RepresentationConfig(coupled_order="source_first"))
        self.assertLess(first.index("Description:"), first.index("Source:"))
        self.assertLess(other.index("Source:"), other.index("Description:"))
        self.assertEqual(sorted(first.split("\n")), sorted(other.split("\n")))      # same content, different order

    def test_function_without_enrichment_falls_back_to_raw_text(self):
        if not self.plain:
            self.skipTest("fixture has no un-enriched function")
        record = self.plain[0]
        text = coupled_text(record, self.cfg)
        self.assertNotIn("Description:", text)
        self.assertIn("Source:", text)

    def test_every_function_gets_a_chunk_and_length_cap_is_applied(self):
        texts = build_texts(self.corpus, "coupled", self.cfg)
        self.assertEqual(set(texts), set(self.corpus.records))
        capped = build_texts(self.corpus, "coupled", RepresentationConfig(coupled_max_chars=120))
        self.assertTrue(all(len(t) <= 120 + len("\n... (truncated)") for t in capped.values()))


class CoupledConfigTests(unittest.TestCase):
    def base(self, **kw):
        data = {"name": "t", "embedding_model": "hashing", "enriched_json": "e.json", "enrichment_model": "m",
                "retrieval": {"representations": ["coupled"], "dense": True}}
        data.update(kw)
        return data

    def test_coupled_requires_enrichment_file(self):
        data = self.base()
        data["enriched_json"] = None
        with self.assertRaises(ConfigError):
            PipelineConfig.from_dict(data)

    def test_invalid_order_is_rejected_and_order_changes_the_hash(self):
        with self.assertRaises(ConfigError):
            PipelineConfig.from_dict(self.base(representation={"coupled_order": "random"}))
        a = PipelineConfig.from_dict(self.base())
        b = PipelineConfig.from_dict(self.base(representation={"coupled_order": "source_first"}))
        self.assertNotEqual(a.config_hash(), b.config_hash())

    def test_coupled_fields_do_not_change_hashes_of_configs_that_do_not_use_them(self):
        plain = PipelineConfig.from_dict({"name": "t", "embedding_model": "hashing", "retrieval": {"representations": ["raw"]}})
        tweaked = PipelineConfig.from_dict({"name": "t", "embedding_model": "hashing", "retrieval": {"representations": ["raw"]},
                                            "representation": {"coupled_order": "source_first", "coupled_max_chars": 999}})
        self.assertEqual(plain.config_hash(), tweaked.config_hash())

    def test_plan_group_builds_matched_pairs(self):
        specs = PlanBuilder("b.json", {"e1": "e1.json"}, "e1", ["bge-small"], "bge-small").build(["coupled"])
        variants = [s.tags["variant"] for s in specs]
        self.assertEqual(variants, ["raw_baseline", "separate_dense", "coupled_dense", "coupled_dense_src",
                                    "separate_hybrid", "coupled_hybrid", "coupled_hybrid_src"])
        by = {s.tags["variant"]: s for s in specs}
        self.assertEqual(by["coupled_hybrid"].config.retrieval.representations, ("coupled",))
        self.assertEqual(by["separate_hybrid"].config.retrieval.representations, ("raw", "semantic"))
        self.assertTrue(by["coupled_hybrid"].config.retrieval.bm25)
        self.assertFalse(by["coupled_dense"].config.retrieval.bm25)
        self.assertEqual(by["coupled_dense_src"].config.representation.coupled_order, "source_first")


class CoupledRetrievalTests(unittest.TestCase):
    def test_one_chunk_is_found_by_enrichment_words_and_by_source_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_fixture(Path(tmp))
            corpus = load_corpus(paths["base"], paths["enriched"])
            cfg = PipelineConfig.from_dict({"name": "t", "embedding_model": "hashing", "enriched_json": str(paths["enriched"]),
                                            "enrichment_model": "m", "retrieval": {"representations": ["coupled"], "dense": True, "bm25": True}})
            pipe = ResearchPipeline(cfg, corpus=corpus, encoder=HashingEncoder())
            pipe.build()
            target = next(r for r in corpus.records.values() if r.has_semantic)
            by_enrichment = str(target.semantic.get("purpose"))
            hit = [r.function_id for r in pipe.search(by_enrichment).results]
            self.assertIn(target.id, hit[:3])
            hit = [r.function_id for r in pipe.search(f"where is {target.name} defined").results]
            self.assertIn(target.id, hit[:3])
            result = pipe.search(by_enrichment)
            self.assertTrue(all(c["source"].endswith(":coupled") for r in result.results for c in r.sources))


if __name__ == "__main__":
    unittest.main()
