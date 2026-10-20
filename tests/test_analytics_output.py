import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analytics_fixtures import build_store
from optima_analytics import plots as plots_mod
from optima_analytics.cli import main
from optima_analytics.pipeline import Config, analyse


def hash_tree(root: Path):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


class OutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.store = build_store(cls.root)
        cls.out = cls.root / "report"
        cls.args = ["--input-dir", str(cls.root / "experiments"), "--enriched-dir", str(cls.store["enr_dir"]),
                    "--base-dir", str(cls.store["base_dir"]), "--output-dir", str(cls.out)]
        cls.before = hash_tree(cls.root / "experiments") | hash_tree(cls.store["enr_dir"]) | hash_tree(cls.store["base_dir"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_1_validate_writes_manifest_and_quality_report_only(self):
        out = self.root / "validate_only"
        code = main(["validate", "--input-dir", str(self.root / "experiments"), "--enriched-dir", str(self.store["enr_dir"]),
                     "--base-dir", str(self.store["base_dir"]), "--output-dir", str(out)])
        self.assertEqual(code, 0)
        self.assertEqual(sorted(p.name for p in out.iterdir() if p.is_file()), ["data_quality_report.md", "experiment_manifest.json"])
        report = (out / "data_quality_report.md").read_text()
        self.assertIn("enrichment_file_misplaced", report)
        self.assertIn("enrichment_base_mismatch", report)

    def test_2_strict_exit_code_on_errors(self):
        out = self.root / "strict"
        code = main(["validate", "--strict", "--input-dir", str(self.root / "experiments"), "--enriched-dir", str(self.store["enr_dir"]),
                     "--base-dir", str(self.store["base_dir"]), "--output-dir", str(out)])
        self.assertEqual(code, 1)

    def test_3_generate_full_report(self):
        code = main(["generate", "--n-boot", "200", "--n-perm", "400"] + self.args)
        self.assertEqual(code, 0)
        for name in ("index.html", "executive_summary.md", "full_analysis.md", "data_quality_report.md", "experiment_manifest.json", "summary_metrics.csv",
                     "model_wise_summary.csv", "category_wise_summary.csv", "pipeline_comparison.csv", "enrichment_gains.csv", "statistical_tests.csv", "run_metadata.json"):
            self.assertTrue((self.out / name).is_file(), name)
        self.assertTrue(list((self.out / "plots").glob("*.png")))
        self.assertTrue(list((self.out / "plot_data").glob("*.csv")))
        self.assertTrue(list((self.out / "models" / "enrichment").glob("*.md")))
        manifest = json.loads((self.out / "experiment_manifest.json").read_text())
        self.assertEqual(manifest["n_runs"], len(self.store["ids"]))
        flags = {r["experiment_id"]: r for r in manifest["runs"]}
        self.assertFalse(flags["e-wrong"]["status_detail"]["analysis_eligible"])
        self.assertFalse(flags["e-base-emb-a-dup"]["replication"]["is_canonical"])
        meta = json.loads((self.out / "run_metadata.json").read_text())
        self.assertEqual(meta["statistics"]["confirmatory_metric"], "mrr")
        self.assertTrue(meta["input_sha256"])

    def test_4_generate_refuses_overwrite_without_force(self):
        self.assertEqual(main(["generate"] + self.args), 2)

    def test_5_output_inside_input_is_refused(self):
        bad = ["generate", "--input-dir", str(self.root / "experiments"), "--output-dir", str(self.root / "experiments" / "x")]
        self.assertEqual(main(bad), 2)

    def test_6_inputs_are_never_modified(self):
        after = hash_tree(self.root / "experiments") | hash_tree(self.store["enr_dir"]) | hash_tree(self.store["base_dir"])
        self.assertEqual(self.before, after)

    def test_7_csv_matches_dashboard_and_manifest_counts(self):
        csv = pd.read_csv(self.out / "summary_metrics.csv")
        self.assertEqual(len(csv), len(self.store["ids"]))
        html = (self.out / "index.html").read_text()
        self.assertNotIn("http://", html.replace("http://www.w3.org", ""))
        self.assertNotIn("https://", html)
        self.assertIn("class=\"sortable\"", html)

    def test_8_every_plot_has_data_and_skipped_have_reasons(self):
        recs = json.loads((self.out / "plot_data" / "_index.json").read_text())
        self.assertTrue(recs)
        for r in recs:
            if r["skipped_reason"]:
                self.assertTrue(len(r["skipped_reason"]) > 10, r["name"])
            else:
                self.assertTrue((self.out / r["png"]).is_file())
                data = pd.read_csv(self.out / r["data_file"])
                self.assertEqual(len(data), r["n_rows"])
                self.assertGreater(len(data), 0)

    def test_9_plot_input_equals_table_values(self):
        res = analyse(Config(input_dirs=[self.root / "experiments"], enriched_dirs=[self.store["enr_dir"]], base_dir=self.store["base_dir"],
                             output_dir=self.root / "r2", n_boot=100, n_perm=100))
        ctx = plots_mod.Context(res)
        plots_mod.style()
        fig, frame, _ = plots_mod.fig_ranked_bars(ctx, "mrr")
        runs = res.data.eligible().set_index("experiment_id")
        for _, r in frame.iterrows():
            source = runs[(runs["dataset"] == r["dataset"]) & (runs["cell"] == r["cell"]) & (runs["embedding_alias"] == r["embedding_alias"])]
            self.assertAlmostEqual(float(source["mrr"].iloc[0]), float(r["mrr"]))
        # raw rows are never labelled with an enrichment model
        self.assertTrue(frame[frame["cell"] == "raw_baseline"]["enrichment_display"].isna().all())

    def test_10_plots_command_subset(self):
        out = self.root / "plots_only"
        code = main(["plots", "--only", "A06", "--n-boot", "100", "--n-perm", "100", "--input-dir", str(self.root / "experiments"),
                     "--enriched-dir", str(self.store["enr_dir"]), "--base-dir", str(self.store["base_dir"]), "--output-dir", str(out)])
        self.assertEqual(code, 0)
        self.assertTrue((out / "plots" / "A06_metric_correlation_matrix.png").is_file())
        self.assertFalse((out / "plots" / "A01_ranked_mrr.png").exists())


if __name__ == "__main__":
    unittest.main()
