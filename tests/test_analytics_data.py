import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analytics_fixtures import BASELINE_RR, build_store, config, write_artifact, write_base, write_run
from optima.rag.research.plan import enriched_base_mismatch
from optima_analytics import ingest
from optima_analytics.catalog import Catalog, default_catalog, size_bucket
from optima_analytics.frame import build_data, classify_cell, run_row
from optima_analytics.schema import IssueLog


def build(store, **kw):
    return build_data([store["root"] / "experiments"], [store["enr_dir"]], store["base_dir"], cache_dir=None, **kw)


class CatalogTests(unittest.TestCase):
    def test_alias_and_dataset_suffix_normalise_to_one_model_and_preserve_original(self):
        c = default_catalog()
        a = c.lookup("enrichment", "Qwen/Qwen2.5-14B-Instruct", "qwen-qwen2.5-14b-instruct-chess")
        b = c.lookup("enrichment", None, "qwen-qwen2.5-14b-instruct")
        self.assertEqual(a.model_id, b.model_id)
        self.assertEqual(a.original, "qwen-qwen2.5-14b-instruct-chess")
        self.assertTrue(a.known)

    def test_general_and_coder_models_are_different(self):
        c = default_catalog()
        self.assertNotEqual(c.lookup("enrichment", None, "qwen-qwen2.5-14b-instruct").model_id,
                            c.lookup("enrichment", None, "qwen-qwen2.5-coder-14b-instruct").model_id)

    def test_unknown_model_is_marked_unknown_not_guessed(self):
        info = default_catalog().lookup("enrichment", "someorg/Mystery-70B")
        self.assertFalse(info.known)
        self.assertIsNone(info.params_b)
        self.assertEqual(info.size_bucket, "unknown")

    def test_size_bucket_boundaries(self):
        self.assertEqual(size_bucket("enrichment", 3.99), "small")
        self.assertEqual(size_bucket("enrichment", 4.0), "medium")
        self.assertEqual(size_bucket("enrichment", 10.0), "large")
        self.assertEqual(size_bucket("embedding", 0.0334), "small")
        self.assertEqual(size_bucket("embedding", 0.109), "medium")
        self.assertEqual(size_bucket("embedding", 0.596), "large")
        self.assertEqual(size_bucket("embedding", None), "unknown")


class IngestTests(unittest.TestCase):
    def test_bad_json_and_missing_fields_are_reported_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "s" / "runs" / "r1").mkdir(parents=True)
            (root / "s" / "runs" / "r1" / "result.json").write_text("{not json")
            (root / "s" / "runs" / "r2").mkdir(parents=True)
            (root / "s" / "runs" / "r2" / "result.json").write_text(json.dumps({"experiment_id": "r2"}))
            issues = IssueLog()
            runs = ingest.load_runs(ingest.discover_run_files([root]), issues)
            self.assertEqual(runs, [])
            self.assertEqual(sorted(i.code for i in issues.items), ["run_missing_fields", "unreadable_json"])

    def test_duplicate_experiment_id_keeps_first_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for s in ("a", "b"):
                write_run(root / s, "same-id", "x", config("baseline_rag", "emb-a"), BASELINE_RR)
            issues = IssueLog()
            runs = ingest.load_runs(ingest.discover_run_files([root]), issues)
            self.assertEqual(len(runs), 1)
            self.assertIn("duplicate_experiment_id", [i.code for i in issues.items])

    def test_missing_per_query_is_a_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = write_run(Path(tmp), "r", "x", config("baseline_rag", "emb-a"), BASELINE_RR, with_per_query=False)
            issues = IssueLog()
            self.assertEqual(ingest.load_per_query(d, "r", issues), [])
            self.assertEqual(issues.items[0].code, "per_query_missing")

    def test_dataset_key(self):
        self.assertEqual(ingest.dataset_key("Chess-Engine-in-cpp-master"), "chess")
        self.assertEqual(ingest.dataset_key("xxhash-dev"), "xxhash")
        self.assertEqual(ingest.dataset_key("cJSON-master"), "cjson")


class FrameTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = build_store(Path(self.tmp.name))
        self.data = build(self.store)

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_runs_kept_failed_run_flagged_not_dropped(self):
        runs = self.data.runs
        self.assertEqual(len(runs), len(self.store["ids"]))
        failed = runs[runs["experiment_id"] == "e-failed"].iloc[0]
        self.assertFalse(failed["ok"])
        self.assertFalse(failed["analysis_eligible"])
        self.assertIn("run_failed", [i.code for i in self.data.issues.items])

    def test_identical_repeat_is_deduplicated_and_noted(self):
        runs = self.data.runs.set_index("experiment_id")
        self.assertFalse(runs.loc["e-base-emb-a-dup", "is_canonical"])
        self.assertEqual(runs.loc["e-base-emb-a-dup", "canonical_id"], "e-base-emb-a")
        self.assertIn("identical_replicates", [i.code for i in self.data.issues.items])

    def test_runs_with_same_config_hash_but_different_enrichment_are_not_merged(self):
        # config_hash ignores enriched_json/enrichment_model; identity must not.
        runs = self.data.runs.set_index("experiment_id")
        self.assertEqual(runs.loc["e-sem-emb-b", "config_hash"], runs.loc["e-wrong", "config_hash"])
        self.assertTrue(runs.loc["e-sem-emb-b", "is_canonical"])
        self.assertTrue(runs.loc["e-wrong", "is_canonical"])

    def test_raw_runs_without_enrichment_are_not_merged_across_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp) / "s"
            for ds in ("chess", "cjson"):
                write_run(suite, f"r-{ds}", f"b|{ds}", config("baseline_rag", "emb-a", base_json=f"{ds}.json"), BASELINE_RR, dataset=ds, chash="same")
            data = build_data([Path(tmp)], [], None)
            self.assertTrue(data.runs["is_canonical"].all())

    def test_misplaced_enrichment_file_excludes_run_and_is_reported(self):
        runs = self.data.runs.set_index("experiment_id")
        self.assertFalse(runs.loc["e-wrong", "analysis_eligible"])
        self.assertTrue(runs.loc["e-sem-emb-a", "analysis_eligible"])
        codes = [i.code for i in self.data.issues.items]
        self.assertIn("enrichment_base_mismatch", codes)
        wrong = self.data.artifacts[self.data.artifacts["artifact"].str.endswith("enr-wrong")].iloc[0]
        self.assertFalse(wrong["base_known"])

    def test_misplaced_artifact_is_flagged_in_the_artifact_table(self):
        self.assertIn("enrichment_file_misplaced", [i.code for i in self.data.issues.items])
        arts = self.data.artifacts.set_index("artifact")
        self.assertTrue(arts.loc["enriched:enr-wrong", "misplaced"])
        self.assertFalse(arts.loc["enriched:enr-good", "misplaced"])

    def test_enrichment_model_identity_uses_model_id_and_keeps_slug(self):
        r = self.data.runs.set_index("experiment_id").loc["e-sem-emb-a"]
        self.assertEqual(r["enrichment_model_id"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(r["enrichment_slug"], "enr-good")
        self.assertEqual(r["enrichment_size"], "medium")

    def test_cell_classification_is_structural(self):
        runs = self.data.runs.set_index("experiment_id")
        self.assertEqual(runs.loc["e-base-emb-a", "cell"], "raw_baseline")
        self.assertEqual(runs.loc["e-rawh-emb-a", "cell"], "raw_hybrid")
        self.assertEqual(runs.loc["e-rawc-emb-a", "cell"], "raw+compiler_hybrid")
        self.assertEqual(runs.loc["e-sem-emb-a", "cell"], "semantic_dense")
        self.assertEqual(runs.loc["e-comp-emb-a", "cell"], "compiler_hybrid")
        self.assertEqual(runs.loc["e-bm25", "cell"], "other")

    def test_non_default_hyperparameters_are_not_in_design_cells(self):
        base = dict(representations="raw", dense=True, bm25=True, exact=False, graph=False, graph_relations="", graph_depth=None,
                    graph_weight=None, rrf_k=10, rerank_model=None)
        self.assertEqual(classify_cell(base), "other")

    def test_per_query_rows_loaded_for_canonical_runs_only(self):
        pq = self.data.per_query
        self.assertNotIn("e-base-emb-a-dup", set(pq["experiment_id"]))
        self.assertEqual(len(pq[pq["experiment_id"] == "e-base-emb-a"]), 12)
        self.assertEqual(set(pq["family"]), {"lexical", "semantic", "structural"})

    def test_artifact_table_has_unique_ids_and_flags(self):
        arts = self.data.artifacts
        self.assertEqual(arts["artifact"].is_unique, True)
        good = arts[arts["artifact"].str.endswith("enr-good")].iloc[0]
        self.assertEqual(good["dataset"], "toy")
        self.assertAlmostEqual(good["core_usable_rate"], 1.0)

    def test_unusable_enrichment_text_is_not_counted_as_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "enhanced_x.json"
            write_artifact(p, "Qwen/Qwen2.5-7B-Instruct", "a" * 64, usable=False)
            item = ingest.load_enrichment_artifact(p, IssueLog())
            self.assertFalse(item["functions"]["core_usable"].any())   # 'behavior' only is not core? purpose/summary empty, behavior present


class PlanGuardTests(unittest.TestCase):
    def test_enriched_file_from_another_base_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "enhanced_x.json"
            write_artifact(p, "m", "f" * 64, dataset="xxhash-dev")
            reason = enriched_base_mismatch(p, "c" * 64)
            self.assertIn("different base.json", reason)
            self.assertIn("xxhash-dev", reason)
            self.assertIsNone(enriched_base_mismatch(p, "f" * 64))

    def test_file_without_recorded_hash_is_accepted_and_unreadable_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "enhanced_y.json"
            p.write_text(json.dumps({"experiment": {}}))
            self.assertIsNone(enriched_base_mismatch(p, "c" * 64))
            p.write_text("{bad")
            self.assertIn("unreadable", enriched_base_mismatch(p, "c" * 64))

    def test_plan_grid_group_builds_matched_counterparts(self):
        from optima.rag.research.plan import PlanBuilder
        b = PlanBuilder("base.json", {"m1": "e1.json", "m2": "e2.json"}, "m1", ["emb-a", "emb-b"], "emb-a")
        specs = b.build(["grid"])
        labels = [s.label for s in specs]
        self.assertEqual(len(labels), len(set(labels)))
        # per embedding: 3 no-enrichment cells + 2 models x 3 enriched cells
        self.assertEqual(len(specs), 2 * (3 + 2 * 3))
        sem = [s for s in specs if s.tags.get("variant") == "semantic_hybrid"][0]
        self.assertEqual(sem.tags["counterpart"], "raw_hybrid")


if __name__ == "__main__":
    unittest.main()
