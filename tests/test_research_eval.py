import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.metrics.retrieval import query_metrics
from optima.rag.research.benchmark import build_structural_benchmark, load_benchmark
from optima.rag.research.config import (NAMED_PIPELINES, ConfigError, PipelineConfig, baseline_rag,
                                        compiler_hybrid_rag, load_config_file, semantic_enrichment_rag)
from optima.rag.research.corpus import load_corpus
from optima.rag.research.dense import HashingEncoder
from optima.rag.research.experiment import ExperimentStore, run_pipeline
from optima.rag.research.metrics import (bootstrap_mean_ci, by_category, evaluate_query, mean_metrics,
                                         paired_bootstrap, relevance_map)
from optima.rag.research.plan import ALL_GROUPS, PlanBuilder
from optima.rag.research.pipeline import ResearchPipeline

from research_fixtures import make_id, write_fixture


class MetricTests(unittest.TestCase):
    def test_binary_metrics_against_hand_computed_values(self):
        ranked = ["x", "a", "y", "b", "z"]
        m = evaluate_query(ranked, {"a": 1.0, "b": 1.0, "c": 1.0})
        self.assertEqual(m["first_relevant_rank"], 2)
        self.assertAlmostEqual(m["mrr"], 0.5)
        self.assertAlmostEqual(m["recall_at_1"], 0.0)
        self.assertAlmostEqual(m["recall_at_3"], 1 / 3)
        self.assertAlmostEqual(m["recall_at_5"], 2 / 3)
        self.assertAlmostEqual(m["precision_at_3"], 1 / 3)
        self.assertAlmostEqual(m["precision_at_5"], 2 / 5)
        self.assertEqual(m["hit_at_1"], 0.0)
        self.assertEqual(m["hit_at_3"], 1.0)
        dcg = 1 / math.log2(3) + 1 / math.log2(5)
        idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)       # three relevant items exist
        self.assertAlmostEqual(m["ndcg_at_5"], dcg / idcg)

    def test_ndcg_ideal_comes_from_judgments_not_from_what_was_retrieved(self):
        # Only one of three relevant items is retrieved, at rank 1.  The old definition
        # (ideal built from retrieved gains) gave exactly 1.0 here.
        m = evaluate_query(["a", "x", "y"], {"a": 1.0, "b": 1.0, "c": 1.0})
        self.assertLess(m["ndcg_at_3"], 1.0)
        self.assertAlmostEqual(m["ndcg_at_3"], 1 / (1 + 1 / math.log2(3) + 0.5))
        legacy = query_metrics([{"function_id": i} for i in ["a", "x", "y"]], {"a", "b", "c"})
        self.assertAlmostEqual(legacy["ndcg_at_3"], m["ndcg_at_3"])    # shared implementation

    def test_graded_ndcg_uses_grades_but_recall_uses_grade_above_zero(self):
        rel = {"a": 2.0, "b": 1.0}
        best = evaluate_query(["a", "b"], rel)
        swapped = evaluate_query(["b", "a"], rel)
        self.assertAlmostEqual(best["ndcg_at_2"] if "ndcg_at_2" in best else best["ndcg_at_3"], 1.0)
        self.assertLess(swapped["ndcg_at_3"], 1.0)
        self.assertEqual(best["recall_at_3"], swapped["recall_at_3"])

    def test_duplicates_in_ranking_count_once_and_short_lists_are_ok(self):
        m = evaluate_query(["a", "a", "a"], {"a": 1.0})
        self.assertEqual(m["precision_at_3"], 1 / 3)              # one unique doc, k=3
        self.assertEqual(evaluate_query([], {"a": 1.0})["mrr"], 0.0)

    def test_queries_without_relevant_documents_are_excluded_not_zero(self):
        self.assertIsNone(evaluate_query(["a"], {}))
        self.assertIsNone(evaluate_query(["a"], {"a": 0.0}))
        rows = [evaluate_query(["a"], {"a": 1.0}), evaluate_query(["b"], {"a": 1.0})]
        self.assertAlmostEqual(mean_metrics(rows)["mrr"], 0.5)
        self.assertEqual(mean_metrics([])["mrr"], None)

    def test_relevance_map_reads_graded_and_legacy_formats(self):
        self.assertEqual(relevance_map({"relevance": {"a": 2, "b": 0}}), {"a": 2.0})
        self.assertEqual(relevance_map({"relevant_functions": ["a", "b"]}), {"a": 1.0, "b": 1.0})
        self.assertEqual(relevance_map({}), {})

    def test_by_category(self):
        rows = [evaluate_query(["a"], {"a": 1.0}), evaluate_query(["x"], {"a": 1.0})]
        out = by_category(rows, ["c1", "c2"])
        self.assertEqual(out["c1"]["mrr"], 1.0)
        self.assertEqual(out["c2"]["mrr"], 0.0)

    def test_bootstrap_ci_covers_mean_and_paired_difference_detects_a_real_gap(self):
        rng = np.random.default_rng(0)
        base = rng.uniform(0, 1, 120)
        better = np.clip(base + 0.2, 0, 1)
        res = paired_bootstrap(better, base, n_boot=2000, seed=1)
        self.assertTrue(res["ci_excludes_zero"])
        self.assertGreater(res["mean_difference"], 0.1)
        self.assertEqual(res["losses"], 0)
        null = paired_bootstrap(base, base, n_boot=500)
        self.assertFalse(null["ci_excludes_zero"])
        self.assertEqual(null["ties"], 120)
        ci = bootstrap_mean_ci(base, n_boot=1000)
        self.assertLess(ci["ci_low"], ci["mean"])
        self.assertGreater(ci["ci_high"], ci["mean"])
        self.assertEqual(bootstrap_mean_ci([])["n"], 0)
        with self.assertRaises(ValueError):
            paired_bootstrap([1.0], [1.0, 2.0])

    def test_cluster_bootstrap_is_wider_than_naive_for_correlated_queries(self):
        values = [1.0] * 10 + [0.0] * 10
        naive = bootstrap_mean_ci(values, n_boot=3000, seed=2)
        clustered = bootstrap_mean_ci(values, groups=[0] * 10 + [1] * 10, n_boot=3000, seed=2)
        self.assertGreater(clustered["ci_high"] - clustered["ci_low"],
                           naive["ci_high"] - naive["ci_low"])


class ConfigTests(unittest.TestCase):
    def test_round_trip_and_hash_stability(self):
        config = compiler_hybrid_rag("b.json", "e.json", embedding_model="bge-small")
        again = PipelineConfig.from_dict(json.loads(json.dumps(config.to_dict())))
        self.assertEqual(config, again)
        self.assertEqual(config.config_hash(), again.config_hash())
        changed = config.replaced({"graph.max_depth": 2})
        self.assertNotEqual(config.config_hash(), changed.config_hash())
        # names, tags and file paths do not affect the functional hash
        moved = PipelineConfig.from_dict({**config.to_dict(), "base_json": "other.json", "name": "n2"})
        self.assertEqual(config.config_hash(), moved.config_hash())

    def test_validation_errors_are_actionable(self):
        bad_cases = [
            ({"name": "x", "bogus": 1}, "unknown pipeline config key"),
            ({"name": "x", "retrieval": {"representations": ["nope"]}}, "unknown representation"),
            ({"name": "x", "retrieval": {"dense": False}}, "at least one of"),
            ({"name": "x", "graph": {"enabled": True, "max_depth": 9}}, "bounded"),
            ({"name": "x", "graph": {"relations": ["magic"]}}, "unknown graph relation"),
            ({"name": "x", "retrieval": {"representations": ["semantic"]}}, "enriched_json is not set"),
            ({"name": "x", "rerank": {"enabled": True, "pool_size": 3}, "top_k": 10}, "pool_size"),
            ({"name": "x", "retrieval": {"rrf_k": 0}}, "rrf_k"),
            ({"name": ""}, "name must not be empty"),
            ({"retrieval": {}}, "requires a 'name'"),
            ({"name": "x", "retrieval": {"typo_key": 1}}, "unknown key"),
        ]
        for data, message in bad_cases:
            with self.subTest(data=data):
                with self.assertRaisesRegex(ConfigError, message):
                    PipelineConfig.from_dict(data)

    def test_config_file_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps(baseline_rag("b.json").to_dict()), encoding="utf-8")
            self.assertEqual(load_config_file(path).name, "baseline_rag")
            path.write_text("{oops", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "not valid JSON"):
                load_config_file(path)
            with self.assertRaisesRegex(ConfigError, "cannot read"):
                load_config_file(Path(tmp) / "missing.json")

    def test_baseline_definition_is_frozen(self):
        """baseline_rag = raw representation, dense only.  Changing this changes every comparison."""
        config = baseline_rag("b.json")
        self.assertEqual(config.retrieval.representations, ("raw",))
        self.assertTrue(config.retrieval.dense)
        self.assertFalse(config.retrieval.bm25)
        self.assertFalse(config.retrieval.exact_match)
        self.assertFalse(config.graph.enabled)
        self.assertFalse(config.rerank.enabled)
        self.assertIsNone(config.enriched_json)
        self.assertEqual(config.top_k, 10)
        self.assertEqual(config.config_hash(), "e2ade24b9823")

    def test_named_pipeline_differences(self):
        sem = semantic_enrichment_rag("b.json", "e.json")
        hyb = compiler_hybrid_rag("b.json", "e.json")
        self.assertEqual(sem.retrieval.representations, ("semantic",))
        self.assertFalse(sem.retrieval.bm25)
        self.assertEqual(hyb.retrieval.representations, ("raw", "semantic", "compiler"))
        self.assertTrue(hyb.retrieval.bm25 and hyb.graph.enabled)
        self.assertFalse(hyb.rerank.enabled)                      # reranking is opt-in
        self.assertEqual(set(NAMED_PIPELINES), {"baseline_rag", "semantic_enrichment_rag",
                                                "compiler_hybrid_rag"})


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.paths = write_fixture(self.dir / "data")
        self.corpus = load_corpus(self.paths["base"], self.paths["enriched"])
        bench = build_structural_benchmark(self.corpus, per_category=4)
        # one query with no judgments and one pointing at unknown IDs
        bench["queries"].append({"query_id": "noj", "category": "behavioral", "query": "x",
                                 "relevance": {}, "relevant_functions": []})
        bench["queries"].append({"query_id": "ghost", "category": "behavioral", "query": "y",
                                 "relevance": {"nope": 1.0}, "relevant_functions": ["nope"]})
        (self.dir / "bench.json").write_text(json.dumps(bench), encoding="utf-8")
        self.bench = load_benchmark(self.dir / "bench.json")

    def run_config(self, config):
        return run_pipeline(config, self.bench, corpus=self.corpus, encoder=HashingEncoder())

    def test_result_record_is_complete_and_serialisable(self):
        config = compiler_hybrid_rag(str(self.paths["base"]), str(self.paths["enriched"]),
                                     "fake", embedding_model="hashing")
        result = self.run_config(config)
        store = ExperimentStore(self.dir / "out", "suite")
        run_dir = store.save_run(result)
        loaded = json.loads((run_dir / "result.json").read_text())
        for key in ("experiment_id", "timestamp", "pipeline", "config", "config_hash", "embedding_model",
                    "enrichment_model", "query_set_version", "ground_truth_version", "dataset", "index",
                    "metrics", "timing", "resources", "errors", "enrichment_coverage", "enrichment_cost"):
            self.assertIn(key, loaded)
        self.assertEqual(loaded["config"], config.to_dict())
        self.assertEqual(PipelineConfig.from_dict(loaded["config"]).config_hash(), loaded["config_hash"])
        self.assertEqual(loaded["enrichment_coverage"]["failed_or_incomplete"], 2)
        skipped = {s["query_id"]: s["reason"] for s in loaded["metrics"]["skipped"]}
        self.assertEqual(skipped["noj"], "no_relevant_judgments")
        self.assertEqual(skipped["ghost"], "relevant_ids_not_in_corpus")
        self.assertEqual(loaded["metrics"]["n_queries_evaluated"], loaded["metrics"]["n_queries_total"] - 2)
        rows = ExperimentStore.load_per_query(run_dir)
        self.assertEqual(len(rows), loaded["metrics"]["n_queries_evaluated"])
        first = rows[0]["ranking"][0]
        for key in ("function_id", "rank", "score", "sources", "representations", "methods"):
            self.assertIn(key, first)
        self.assertIn("recall_after_graph", loaded["metrics"]["candidate_pool"])
        self.assertGreater(loaded["index"]["index_bytes_total"], 0)

    def test_results_are_never_overwritten(self):
        result = self.run_config(baseline_rag(str(self.paths["base"]), embedding_model="hashing"))
        store = ExperimentStore(self.dir / "out", "suite")
        store.save_run(result)
        with self.assertRaises(FileExistsError):
            store.save_run(result)
        second = self.run_config(baseline_rag(str(self.paths["base"]), embedding_model="hashing"))
        self.assertNotEqual(result["experiment_id"], second["experiment_id"])
        store.save_run(second)
        self.assertEqual(len(store.load_runs()), 2)

    def test_running_the_same_config_twice_gives_identical_rankings(self):
        config = baseline_rag(str(self.paths["base"]), embedding_model="hashing")
        a, b = self.run_config(config), self.run_config(config)
        self.assertEqual(a["metrics"]["overall"], b["metrics"]["overall"])
        self.assertEqual([[r["function_id"] for r in q["ranking"]] for q in a["_per_query"]],
                         [[r["function_id"] for r in q["ranking"]] for q in b["_per_query"]])

    def test_enabling_an_advanced_stage_does_not_change_the_baseline(self):
        base = baseline_rag(str(self.paths["base"]), embedding_model="hashing")
        before = self.run_config(base)
        # build and run a different, fully-advanced pipeline in the same process ...
        self.run_config(compiler_hybrid_rag(str(self.paths["base"]), str(self.paths["enriched"]),
                                            embedding_model="hashing"))
        after = self.run_config(base)                              # ... then the baseline again
        self.assertEqual(before["metrics"]["overall"], after["metrics"]["overall"])

    def test_stages_are_independent_switches(self):
        common = dict(base_json=str(self.paths["base"]), enriched_json=str(self.paths["enriched"]),
                      embedding_model="hashing")
        def ranking(**sections):
            cfg = PipelineConfig.from_dict({"name": "t", **common, **sections})
            pipe = ResearchPipeline(cfg, corpus=self.corpus, encoder=HashingEncoder())
            return [i.function_id for i in pipe.search("parse json tokens").results]
        lexical = {"representations": ["raw", "compiler"], "dense": False, "bm25": True}
        plain = ranking(retrieval=lexical)
        self.assertEqual(plain, ranking(retrieval=lexical, graph={"enabled": False}))
        self.assertEqual(plain, ranking(retrieval=lexical, rerank={"enabled": False}))
        # toggling the exact matcher leaves dense/bm25 provenance untouched
        cfg = PipelineConfig.from_dict({"name": "t", **common, "retrieval": {**lexical, "exact_match": True}})
        top = ResearchPipeline(cfg, corpus=self.corpus, encoder=HashingEncoder()).search("parse_json").results[0]
        self.assertIn("exact", top.methods)
        self.assertIn("bm25", top.methods)

    def test_failed_enrichment_is_reported_in_coverage_not_hidden(self):
        result = self.run_config(semantic_enrichment_rag(
            str(self.paths["base"]), str(self.paths["enriched"]), "fake", embedding_model="hashing"))
        coverage = result["enrichment_coverage"]
        self.assertEqual(coverage["status_counts"]["failed"], 1)
        self.assertEqual(coverage["status_counts"]["missing"], 1)
        self.assertAlmostEqual(coverage["semantic_coverage"], 0.75)
        self.assertEqual(result["enrichment_cost"]["total_tokens"], 720)


class PlanTests(unittest.TestCase):
    def test_every_reference_exists_labels_are_unique_and_configs_valid(self):
        builder = PlanBuilder("b.json", {"e1": "e1.json", "e2": "e2.json"}, "e1",
                              ["bge-small", "bge-base"], "bge-base", ["ms-marco-minilm"],
                              routing_profiles={"literal": {"bm25_weight": 4.0}})
        specs = builder.build(ALL_GROUPS)
        labels = [s.label for s in specs]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertTrue(all(s.reference in labels for s in specs if s.reference))
        self.assertGreater(len(specs), 60)

    def test_improved_group_needs_profiles_and_pairs_with_a_routed_counterpart(self):
        with self.assertRaises(ValueError):
            PlanBuilder("b.json", {"e1": "e1.json"}, "e1", ["bge-small"], "bge-small").build(["improved"])
        builder = PlanBuilder("b.json", {"e1": "e1.json"}, "e1", ["bge-small"], "bge-small",
                              routing_profiles={"natural": {"bm25_weight": 0.0}})
        specs = builder.build(["improved"])
        routed = [s for s in specs if s.config.routing.enabled]
        self.assertEqual(len(routed), 2)         # raw+compiler counterpart and the enriched compiler stack
        self.assertTrue(all(s.config.routing.profiles == {"natural": {"bm25_weight": 0.0}} for s in routed))
        enriched = next(s for s in routed if s.tags["variant"] == "compiler_hybrid_routed")
        self.assertEqual(enriched.tags["counterpart"], "raw+compiler_hybrid_routed")

    def test_representation_ablation_covers_all_seven_subsets_in_three_modes(self):
        builder = PlanBuilder("b.json", {"e1": "e1.json"}, "e1", ["bge-small"], "bge-small")
        specs = [s for s in builder.build(["representations"])]
        combos = {s.tags["representations"] for s in specs}
        self.assertEqual(len(combos), 7)
        self.assertEqual({s.tags["mode"] for s in specs}, {"dense", "bm25", "dense+bm25"})
        self.assertEqual(len(specs), 21)

    def test_graph_group_includes_a_no_graph_reference_and_rerank_is_off_everywhere_else(self):
        builder = PlanBuilder("b.json", {"e1": "e1.json"}, "e1", ["bge-small"], "bge-small")
        specs = builder.build(["graph"])
        self.assertTrue(any(s.tags["variant"] == "no_graph" for s in specs))
        self.assertTrue(all(not s.config.rerank.enabled for s in specs))

    def test_unknown_group_and_bad_primary_are_rejected(self):
        builder = PlanBuilder("b.json", {"e1": "e1.json"}, "e1", ["bge-small"], "bge-small")
        with self.assertRaises(ValueError):
            builder.build(["nonsense"])
        with self.assertRaises(ValueError):
            PlanBuilder("b.json", {"e1": "e1.json"}, "zzz", ["bge-small"], "bge-small")


class SuiteEndToEndTests(unittest.TestCase):
    def test_tiny_suite_runs_and_writes_all_artifacts(self):
        from optima.rag.research.suite import run_suite
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = write_fixture(tmp / "data")
            corpus = load_corpus(paths["base"])
            bench = build_structural_benchmark(corpus, per_category=3)
            (tmp / "bench.json").write_text(json.dumps(bench), encoding="utf-8")
            spec = {"suite_id": "tiny", "embedding_models": ["hashing"], "primary_embedding": "hashing",
                    "groups": ["pipelines", "hybrid", "graph"], "bootstrap_samples": 200,
                    "bootstrap_samples_per_run": 100,
                    "datasets": [{"name": "synthetic", "base_json": str(paths["base"]),
                                  "benchmark": str(tmp / "bench.json"),
                                  "enriched": {"fake": str(paths["enriched"])},
                                  "primary_enrichment": "fake"}]}
            outcome = run_suite(spec, tmp / "out")
            suite_dir = Path(outcome["suite_dir"])
            self.assertEqual(outcome["summary"]["failed"], 0, outcome)
            self.assertGreater(outcome["summary"]["reused"], 0)    # identical configs execute once
            for name in ("aggregate.csv", "per_category.csv", "per_query_metrics.csv",
                         "comparisons.csv", "report.md", "suite.json"):
                self.assertTrue((suite_dir / name).is_file(), name)
            report = (suite_dir / "report.md").read_text()
            self.assertIn("Pipeline comparison", report)
            self.assertIn("Ablation: graph", report)
            self.assertIn("Candidate-pool coverage", report)
            before = len(list((suite_dir / "runs").iterdir()))
            run_suite(spec, tmp / "out", resume=True)
            self.assertEqual(len(list((suite_dir / "runs").iterdir())), before)  # nothing re-run
            self.assertTrue(list((suite_dir / "plots").glob("*.png")))

    def test_unavailable_embedding_model_is_recorded_as_failed_not_mocked(self):
        from unittest import mock
        from optima.rag.research.suite import run_suite
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = write_fixture(tmp / "data")
            bench = build_structural_benchmark(load_corpus(paths["base"]), per_category=2)
            (tmp / "bench.json").write_text(json.dumps(bench), encoding="utf-8")
            spec = {"suite_id": "fail", "embedding_models": ["bge-small"], "groups": ["pipelines"],
                    "bootstrap_samples": 50,
                    "datasets": [{"name": "s", "base_json": str(paths["base"]),
                                  "benchmark": str(tmp / "bench.json"),
                                  "enriched": {"fake": str(paths["enriched"])}}]}
            with mock.patch("sentence_transformers.SentenceTransformer", side_effect=OSError("offline")):
                outcome = run_suite(spec, tmp / "out")
            self.assertEqual(outcome["summary"]["failed"], 3)
            report = (Path(outcome["suite_dir"]) / "report.md").read_text()
            self.assertIn("Runs that could not be executed", report)
            self.assertIn("embedding model", report)


if __name__ == "__main__":
    unittest.main()
