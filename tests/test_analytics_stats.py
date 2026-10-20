import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analytics_fixtures import BASELINE_RR, CATEGORIES, QUERY_IDS, SEEDS, build_store
from optima_analytics import stats
from optima_analytics.analysis import Options, enrichment_gains, category_tables, pipeline_comparison
from optima_analytics.diagnostics import difficulty_tiers
from optima_analytics.frame import build_data
from optima_analytics.models import pareto_front, rankings
from optima_analytics.schema import METRIC_DIRECTION
from optima_analytics.scores import Scores


class GainTests(unittest.TestCase):
    def test_absolute_and_relative_gain(self):
        self.assertAlmostEqual(stats.absolute_gain(0.6, 0.5), 0.1)
        self.assertAlmostEqual(stats.relative_gain_pct(0.6, 0.5), 20.0)

    def test_zero_or_missing_baseline_gives_nan_not_inf(self):
        self.assertTrue(math.isnan(stats.relative_gain_pct(0.3, 0.0)))
        self.assertTrue(math.isnan(stats.relative_gain_pct(float("nan"), 0.5)))
        self.assertTrue(math.isnan(stats.relative_gain_pct(0.3, None)))

    def test_lower_is_better_direction_flips_improvement(self):
        self.assertFalse(METRIC_DIRECTION["query_ms_mean"])
        self.assertTrue(METRIC_DIRECTION["mrr"])
        # latency 10ms -> 8ms is an improvement of +20%
        self.assertAlmostEqual(stats.improvement_pct(8.0, 10.0, higher_is_better=False), 20.0)
        self.assertAlmostEqual(stats.improvement_pct(12.0, 10.0, higher_is_better=True), 20.0)


class MultipleComparisonTests(unittest.TestCase):
    def test_holm_matches_hand_computation(self):
        self.assertEqual([round(x, 6) for x in stats.holm_adjust([0.01, 0.04, 0.03, 0.005])], [0.03, 0.06, 0.06, 0.02])

    def test_bh_matches_hand_computation(self):
        self.assertEqual([round(x, 6) for x in stats.bh_adjust([0.01, 0.04, 0.03, 0.005])], [0.02, 0.04, 0.04, 0.02])

    def test_adjustments_are_monotone_capped_and_keep_nan(self):
        out = stats.holm_adjust([0.5, float("nan"), 0.9])
        self.assertTrue(math.isnan(out[1]))
        self.assertTrue(all(v <= 1.0 for v in (out[0], out[2])))

    def test_families_are_corrected_separately(self):
        df = pd.DataFrame({"family": ["a", "a", "b"], "p_value": [0.02, 0.03, 0.02]})
        out = stats.add_adjusted(df)
        self.assertAlmostEqual(out.loc[2, "p_holm"], 0.02)          # family b has one test
        self.assertAlmostEqual(out.loc[0, "p_holm"], 0.04)
        self.assertEqual(list(out["family_size"]), [2, 2, 1])


class PairedTestTests(unittest.TestCase):
    def test_exact_sign_flip_known_value(self):
        res = stats.cluster_sign_flip_p([1, 1, 1, 1], [0, 0, 1, 1])
        self.assertEqual(res["method"], "exact sign-flip")
        self.assertAlmostEqual(res["p_value"], 0.5)

    def test_identical_systems_are_not_significant(self):
        a = np.random.default_rng(0).random(60)
        res = stats.paired_compare(a, a.copy(), groups=list(range(60)))
        self.assertEqual(res["p_value"], 1.0)
        self.assertEqual(res["ties"], 60)
        self.assertFalse(res["ci_excludes_zero"])

    def test_known_effect_is_detected_with_effect_size_and_counts(self):
        rng = np.random.default_rng(1)
        ref = rng.random(80)
        cand = ref + 0.1 + rng.normal(0, 0.02, 80)
        res = stats.paired_compare(cand, ref, groups=list(range(80)), n_boot=2000, n_perm=4000)
        self.assertLess(res["p_value"], 0.01)
        self.assertGreater(res["ci_low"], 0.0)
        self.assertGreater(res["cohen_dz"], 2)
        self.assertEqual(res["wins"] + res["losses"] + res["ties"], 80)
        self.assertAlmostEqual(res["rel_gain_pct"], 100 * (cand.mean() - ref.mean()) / ref.mean())

    def test_null_difference_is_usually_not_significant(self):
        rng = np.random.default_rng(2)
        a, b = rng.random(80), rng.random(80)
        res = stats.paired_compare(a, b, groups=list(range(80)), n_boot=1000, n_perm=2000)
        self.assertGreater(res["p_value"], 0.01)

    def test_clustering_widens_the_interval(self):
        rng = np.random.default_rng(3)
        d = np.repeat(rng.normal(0.05, 0.2, 20), 5)              # 20 clusters of 5 identical differences
        ref = np.zeros(100)
        free = stats.paired_compare(d, ref, groups=list(range(100)), n_boot=2000, n_perm=100)
        clustered = stats.paired_compare(d, ref, groups=list(np.repeat(range(20), 5)), n_boot=2000, n_perm=100)
        self.assertGreater(clustered["ci_high"] - clustered["ci_low"], free["ci_high"] - free["ci_low"])

    def test_nan_pairs_are_dropped_and_length_mismatch_rejected(self):
        res = stats.paired_compare([1.0, float("nan"), 0.5], [0.5, 0.5, 0.5])
        self.assertEqual(res["n"], 2)
        with self.assertRaises(ValueError):
            stats.paired_compare([1, 2], [1])

    def test_empty_input_returns_nans(self):
        res = stats.paired_compare([], [])
        self.assertEqual(res["n"], 0)
        self.assertTrue(math.isnan(res["p_value"]))


class SummaryTests(unittest.TestCase):
    def test_describe_statistics(self):
        d = stats.describe([1, 2, 3, 4, 5, 6])
        self.assertEqual(d["n"], 6)
        self.assertAlmostEqual(d["mean"], 3.5)
        self.assertAlmostEqual(d["median"], 3.5)
        self.assertAlmostEqual(d["iqr"], 2.5)
        self.assertAlmostEqual(d["std"], np.std([1, 2, 3, 4, 5, 6], ddof=1))
        self.assertFalse(math.isnan(d["ci_low"]))

    def test_describe_small_n_has_no_ci_and_empty_is_nan(self):
        self.assertTrue(math.isnan(stats.describe([1, 2, 3])["ci_low"]))
        self.assertEqual(stats.describe([])["n"], 0)

    def test_small_n_correlation_is_flagged_unreliable(self):
        c = stats.correlation(list(range(8)), [x * 2 for x in range(8)])
        self.assertFalse(c["reliable"])
        self.assertAlmostEqual(c["rho"], 1.0)
        self.assertIn("descriptive", c["note"])

    def test_constant_input_correlation_is_undefined(self):
        c = stats.correlation([1, 1, 1, 1], [1, 2, 3, 4])
        self.assertTrue(math.isnan(c["rho"]))

    def test_ci_overlap(self):
        self.assertTrue(stats.ci_overlap(0.1, 0.3, 0.2, 0.4))
        self.assertFalse(stats.ci_overlap(0.1, 0.2, 0.3, 0.4))


class StoreAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.store = build_store(Path(cls.tmp.name))
        cls.data = build_data([cls.store["root"] / "experiments"], [cls.store["enr_dir"]], cls.store["base_dir"])
        cls.scores = Scores(cls.data.per_query)
        cls.opts = Options(n_boot=300, n_perm=500)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_gain_uses_matched_counterpart_and_excludes_misplaced_run(self):
        gains, tests = enrichment_gains(self.data, self.scores, self.opts)
        self.assertNotIn("Qwen/Qwen2.5-3B-Instruct", set(gains["enrichment_model_id"]))
        g = gains[(gains["metric"] == "mrr") & (gains["cell"] == "semantic_dense") & (gains["embedding"] == "emb-a") & (gains["reference_type"] == "matched_no_enrichment")].iloc[0]
        self.assertAlmostEqual(g["abs_gain"], np.mean([min(1.0, v + 0.04) - v for v in BASELINE_RR]))
        c = gains[(gains["metric"] == "mrr") & (gains["cell"] == "compiler_hybrid") & (gains["embedding"] == "emb-a") & (gains["reference_type"] == "matched_no_enrichment")].iloc[0]
        self.assertEqual(c["reference_cell"], "raw+compiler_hybrid")
        self.assertTrue(gains["higher_is_better"].dropna().all())
        self.assertTrue(len(tests) > 0 and set(tests["metric"]) <= {"mrr", "ndcg_at_10", "recall_at_10"})

    def test_only_primary_metric_is_confirmatory(self):
        _, tests = enrichment_gains(self.data, self.scores, self.opts)
        self.assertTrue(tests[tests["metric"] == "mrr"]["confirmatory"].all())
        self.assertFalse(tests[tests["metric"] != "mrr"]["confirmatory"].any())

    def test_pipeline_comparison_pairs_only_comparable_runs(self):
        table, tests = pipeline_comparison(self.data, self.scores, self.opts)
        self.assertTrue((table["n_queries"] == 12).all())
        self.assertIn("compiler_hybrid_rag vs baseline_rag", set(tests["comparison"]))
        self.assertTrue(table["experiment_id"].is_unique)

    def test_category_table_denominators(self):
        gains, _ = enrichment_gains(self.data, self.scores, self.opts)
        cat, _ = category_tables(self.data, self.scores, gains, self.opts, {"toy": "emb-b"}, {"toy": "enr-good"})
        row = cat[(cat["dimension"] == "category") & (cat["level"] == "behavioral") & (cat["cell"] == "raw_baseline") & (cat["embedding_alias"] == "emb-a")].iloc[0]
        beh = [v for v, c in zip(BASELINE_RR, CATEGORIES) if c == "behavioral"]
        self.assertAlmostEqual(row["mrr"], np.mean(beh))              # mean over that category's queries
        self.assertEqual(row["n_queries"], 4)
        fam = cat[(cat["dimension"] == "family") & (cat["level"] == "lexical") & (cat["cell"] == "raw_baseline") & (cat["embedding_alias"] == "emb-a")].iloc[0]
        self.assertEqual(fam["n_queries"], 4)

    def test_difficulty_tiers_depend_only_on_the_independent_bm25_run(self):
        tiers = difficulty_tiers(self.data, self.scores)
        t = tiers.set_index("query_id")
        self.assertEqual(t.loc["q00", "tier"], "easy")      # BM25 RR 1.0
        self.assertEqual(t.loc["q05", "tier"], "medium")    # BM25 RR 0.3
        self.assertEqual(t.loc["q03", "tier"], "hard")      # BM25 RR 0
        self.assertEqual(set(tiers["reference_run"]), {"e-bm25"})

    def test_significant_losses_are_not_counted_as_significant_gains(self):
        from optima_analytics.models import enrichment_model_summary
        gains, _ = enrichment_gains(self.data, self.scores, self.opts)
        gains = gains.copy()
        m = (gains["metric"] == "mrr") & (gains["cell"] == "semantic_dense") & (gains["reference_type"] == "matched_no_enrichment")
        gains.loc[m, "significant_holm_05"] = True
        gains.loc[m, "abs_gain"] = -0.1
        row = enrichment_model_summary(self.data, gains).iloc[0]
        self.assertEqual(row["semantic_dense__n_gain_sig_vs_matched_no_enrichment"], 0)
        self.assertEqual(row["semantic_dense__n_loss_sig_vs_matched_no_enrichment"], int(m.sum()))
        gains.loc[m, "abs_gain"] = 0.1
        row = enrichment_model_summary(self.data, gains).iloc[0]
        self.assertEqual(row["semantic_dense__n_gain_sig_vs_matched_no_enrichment"], int(m.sum()))
        self.assertEqual(row["semantic_dense__n_loss_sig_vs_matched_no_enrichment"], 0)

    def test_superseded_query_set_is_excluded_from_analyses(self):
        from optima_analytics.frame import flag_superseded
        import pandas as pd
        from optima_analytics.schema import IssueLog
        runs = pd.DataFrame({"dataset": ["d", "d", "e"], "query_set_version": ["1.0.0", "2.0.0", "1.0.0"], "analysis_eligible": [True, True, True]})
        issues = IssueLog()
        out = flag_superseded(runs, issues)
        self.assertEqual(list(out["superseded_query_set"]), [True, False, False])
        self.assertEqual(list(out["analysis_eligible"]), [False, True, True])
        self.assertIn("superseded_query_set", [i.code for i in issues.items])

    def test_pareto_front(self):
        df = pd.DataFrame({"query_ms_mean": [1, 2, 3, 2], "mrr": [0.3, 0.5, 0.4, 0.2]})
        self.assertEqual(list(pareto_front(df, "query_ms_mean", "mrr")), [True, True, False, False])

    def test_rankings_flag_overlap_and_carry_run_counts(self):
        from optima_analytics.analysis import pipeline_comparison as pc
        from optima_analytics.models import enrichment_model_summary, embedding_model_summary
        gains, _ = enrichment_gains(self.data, self.scores, self.opts)
        enr, emb = enrichment_model_summary(self.data, gains), embedding_model_summary(self.data, gains)
        cat, _ = category_tables(self.data, self.scores, gains, self.opts, {"toy": "emb-b"}, {"toy": "enr-good"})
        r = rankings(self.data, gains, enr, emb, cat, pc(self.data, self.scores, self.opts)[0])
        top = r[(r["objective"] == "Best overall retrieval quality (MRR), any configuration") & (r["rank"] == 1)].iloc[0]
        self.assertEqual(top["n_runs"], 1)
        self.assertIn(top["indistinguishable_from_top"], ("yes", "no"))
        self.assertTrue(r["caveat"].notna().all())
        # no composite score anywhere
        self.assertFalse(r["objective"].str.contains("composite|overall score", case=False).any())


if __name__ == "__main__":
    unittest.main()
