import json
import tempfile
import unittest
from pathlib import Path

from optima.rag.research.benchmark import (BenchmarkError, build_structural_benchmark,
                                           load_benchmark, validate_benchmark)
from optima.rag.research.config import RepresentationConfig
from optima.rag.research.corpus import CorpusError, load_corpus
from optima.rag.research.representations import build_texts

from research_fixtures import make_id, write_fixture


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_loads_raw_and_enriched_and_preserves_ids(self):
        paths = write_fixture(self.dir)
        raw = load_corpus(paths["base"])
        enriched = load_corpus(paths["base"], paths["enriched"])
        self.assertEqual(len(raw), 8)
        self.assertEqual(raw.ids, enriched.ids)                  # same universe, same order
        self.assertIn(make_id("parse"), raw.records)             # IDs are verbatim, never rewritten
        self.assertEqual(raw.records[make_id("parse")].enrichment_status, "missing")
        self.assertEqual(enriched.records[make_id("parse")].enrichment_status, "ok")
        self.assertEqual(enriched.records[make_id("parse")].semantic["purpose"],
                         "Parse a JSON document into a tree")

    def test_missing_failed_and_empty_enrichment_keep_the_function(self):
        paths = write_fixture(self.dir)
        corpus = load_corpus(paths["base"], paths["enriched"])
        self.assertEqual(corpus.records[make_id("hash")].enrichment_status, "failed")
        self.assertEqual(corpus.records[make_id("read2")].enrichment_status, "missing")
        self.assertEqual(len(corpus), 8)
        summary = corpus.meta["enrichment"]
        self.assertEqual(summary["status_counts"]["ok"], 6)
        self.assertEqual(summary["failed_or_incomplete"], 2)
        self.assertAlmostEqual(summary["semantic_coverage"], 6 / 8)
        self.assertEqual(summary["usage_totals"]["total_tokens"], 6 * 120)

    def test_duplicate_ids_keep_first_and_are_counted(self):
        paths = write_fixture(self.dir, duplicate_base=True, include_duplicate=True, extra_id=True)
        corpus = load_corpus(paths["base"], paths["enriched"])
        self.assertEqual(len(corpus), 8)
        self.assertEqual(corpus.meta["duplicate_ids"], 1)
        # the enriched file inherits the base duplicate and adds one of its own
        self.assertEqual(corpus.meta["enriched_duplicate_ids"], 2)
        self.assertEqual(corpus.meta["enriched_only_ids"], 1)
        self.assertNotIn("src/stray.c::stray::1", corpus.records)

    def test_invalid_json_and_wrong_shape_give_actionable_errors(self):
        bad = self.dir / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with self.assertRaisesRegex(CorpusError, "not valid JSON"):
            load_corpus(bad)
        wrong = self.dir / "wrong.json"
        wrong.write_text(json.dumps({"hello": 1}), encoding="utf-8")
        with self.assertRaisesRegex(CorpusError, "'files' list"):
            load_corpus(wrong)
        with self.assertRaisesRegex(CorpusError, "not found"):
            load_corpus(self.dir / "missing.json")

    def test_cfg_summary_detects_loops(self):
        corpus = load_corpus(write_fixture(self.dir)["base"])
        self.assertTrue(corpus.records[make_id("read")].cfg["has_loop"])
        self.assertFalse(corpus.records[make_id("parse")].cfg["has_loop"])


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        paths = write_fixture(Path(self._tmp.name))
        self.corpus = load_corpus(paths["base"], paths["enriched"])
        self.cfg = RepresentationConfig()

    def test_raw_has_source_semantic_has_enrichment_compiler_has_structure(self):
        pid = make_id("parse")
        raw = build_texts(self.corpus, "raw", self.cfg)[pid]
        sem = build_texts(self.corpus, "semantic", self.cfg)[pid]
        comp = build_texts(self.corpus, "compiler", self.cfg)[pid]
        self.assertIn("alloc_node();", raw)
        self.assertNotIn("Purpose", raw)
        self.assertIn("Purpose: Parse a JSON document into a tree", sem)
        self.assertNotIn("alloc_node();", sem)
        self.assertIn("Calls: read_token, alloc_node", comp)
        self.assertNotIn("return 0;", comp)

    def test_functions_with_failed_enrichment_are_not_dropped_by_default(self):
        texts = build_texts(self.corpus, "semantic", self.cfg)
        self.assertEqual(len(texts), 8)
        self.assertNotIn("Purpose", texts[make_id("hash")])
        self.assertIn("hash_string", texts[make_id("hash")])

    def test_missing_semantic_policies(self):
        excl = build_texts(self.corpus, "semantic", RepresentationConfig(missing_semantic="exclude"))
        self.assertEqual(len(excl), 6)
        self.assertNotIn(make_id("hash"), excl)
        raw_fallback = build_texts(self.corpus, "semantic", RepresentationConfig(missing_semantic="raw"))
        self.assertIn("5381", raw_fallback[make_id("hash")])

    def test_unmentioned_large_blobs_are_not_rendered(self):
        text = build_texts(self.corpus, "compiler", self.cfg)[make_id("read")]
        self.assertIn("contains a loop", text)
        self.assertLess(len(text), 400)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.corpus = load_corpus(write_fixture(Path(self._tmp.name))["base"])

    def test_exact_name_labels_include_every_function_with_that_name(self):
        bench = build_structural_benchmark(self.corpus, per_category=20)
        exact = [q for q in bench["queries"] if q["category"] == "exact_name"
                 and "read_token" in q["query"]]
        self.assertEqual(len(exact), 1)
        self.assertEqual(set(exact[0]["relevance"]), {make_id("read"), make_id("read2")})

    def test_caller_and_callee_labels_follow_the_call_graph(self):
        bench = build_structural_benchmark(self.corpus, per_category=20)
        callers = {q["query"]: q for q in bench["queries"] if q["category"] == "caller"}
        target = next(q for text, q in callers.items() if "is_digit" in text)
        self.assertEqual(set(target["relevance"]), {make_id("read")})
        callees = [q for q in bench["queries"] if q["category"] == "callee" and "parse_json" in q["query"]]
        self.assertEqual(set(callees[0]["relevance"]), {make_id("read"), make_id("alloc")})

    def test_multi_hop_is_graded_and_dependency_category_is_declared_unsupported(self):
        bench = build_structural_benchmark(self.corpus, per_category=20)
        self.assertIn("dependency", bench["unsupported_categories"])
        hops = [q for q in bench["queries"] if q["category"] == "multi_hop" and "is_digit" in q["query"]]
        self.assertTrue(hops)
        self.assertEqual(hops[0]["relevance"][make_id("read")], 2.0)
        self.assertEqual(hops[0]["relevance"][make_id("parse")], 1.0)

    def test_labels_are_deterministic_and_validate(self):
        a = build_structural_benchmark(self.corpus, per_category=5, seed=3)
        b = build_structural_benchmark(self.corpus, per_category=5, seed=3)
        self.assertEqual([q["query"] for q in a["queries"]], [q["query"] for q in b["queries"]])
        path = Path(self._tmp.name) / "bench.json"
        path.write_text(json.dumps(a), encoding="utf-8")
        report = validate_benchmark(load_benchmark(path), self.corpus)
        self.assertTrue(report["ok"], report)

    def test_authored_queries_resolve_by_file_and_qualified_name_or_fail_loudly(self):
        good = {"queries": [{"query": "check digit", "targets": [
            {"file": "digit.c", "qualified_name": "is_digit", "grade": 2}]}]}
        bench = build_structural_benchmark(self.corpus, per_category=1, authored=good)
        self.assertEqual([q for q in bench["queries"] if q["category"] == "behavioral"][0]["relevance"],
                         {make_id("digit"): 2.0})
        bad = {"queries": [{"query": "x", "targets": [{"file": "nope", "qualified_name": "nothing"}]}]}
        with self.assertRaises(BenchmarkError):
            build_structural_benchmark(self.corpus, per_category=1, authored=bad)

    def test_validation_flags_unknown_ids_empty_judgments_and_duplicates(self):
        data = {"benchmark_id": "x", "query_set_version": "1", "ground_truth_version": "1",
                "project": "p", "base_sha256": "", "queries": [
                    {"query_id": "a", "query": "q1", "relevant_functions": ["nope"]},
                    {"query_id": "a", "query": "q2", "relevant_functions": []}]}
        path = Path(self._tmp.name) / "b.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        report = validate_benchmark(load_benchmark(path), self.corpus)
        self.assertFalse(report["ok"])
        self.assertEqual(report["queries_with_invalid_ids"], 1)
        self.assertEqual(report["queries_without_judgments"], ["a"])
        self.assertTrue(report["duplicates"])

    def test_legacy_list_format_loads(self):
        path = Path(self._tmp.name) / "legacy.json"
        path.write_text(json.dumps([{"query": "q", "relevant_functions": [make_id("digit")]}]),
                        encoding="utf-8")
        bench = load_benchmark(path)
        self.assertEqual(bench.query_set_version, "legacy")
        self.assertEqual(bench.queries[0]["query_id"], "q_0001")


if __name__ == "__main__":
    unittest.main()
