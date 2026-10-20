import unittest

from optima_analytics.coupled import PAIRS
from optima_analytics.frame import classify_cell, _reps

BASE = dict(representations="raw", dense=True, bm25=False, exact=False, graph=False, graph_relations="",
            graph_depth=None, graph_weight=None, rrf_k=60, rerank_model=None, coupled_order="semantic_first")


def cell(**kw):
    return classify_cell({**BASE, **kw})


class CoupledCellTests(unittest.TestCase):
    def test_coupled_representation_is_recognised(self):
        self.assertEqual(_reps({"retrieval": {"representations": ["coupled"]}}), ["coupled"])

    def test_dense_and_hybrid_cells_and_order_suffix(self):
        self.assertEqual(cell(representations="coupled"), "coupled_dense")
        self.assertEqual(cell(representations="coupled", coupled_order="source_first"), "coupled_dense_src")
        self.assertEqual(cell(representations="coupled", bm25=True, exact=True), "coupled_hybrid")
        self.assertEqual(cell(representations="coupled", bm25=True, exact=True, coupled_order="source_first"), "coupled_hybrid_src")

    def test_separate_cells_are_distinct_from_coupled(self):
        self.assertEqual(cell(representations="raw+semantic"), "separate_dense")
        self.assertEqual(cell(representations="raw+semantic", bm25=True, exact=True), "separate_hybrid")

    def test_non_default_settings_stay_other(self):
        self.assertEqual(cell(representations="coupled", rrf_k=10), "other")
        self.assertEqual(cell(representations="coupled", graph=True, graph_relations="callers", graph_depth=2, graph_weight=0.1), "other")

    def test_routed_raw_plus_semantic_run_is_an_attribution_cell_not_a_separate_cell(self):
        # attribution factorial runs (routing + instruction, raw+semantic) must not be absorbed by the coupled-chunk "separate" cells
        from optima_analytics import frame
        row = {**BASE, "representations": "raw+semantic", "bm25": True, "exact": True, "routing": True, "query_prefix": "x"}
        base = classify_cell(row)
        self.assertEqual(base, "separate_hybrid")
        src = open(frame.__file__).read()
        self.assertIn('("other", "separate_dense", "separate_hybrid") and row["routing"]', src)

    def test_every_pair_refers_to_a_real_cell(self):
        known = {"coupled_dense", "coupled_dense_src", "coupled_hybrid", "coupled_hybrid_src", "separate_dense",
                 "separate_hybrid", "semantic_dense", "raw_baseline"}
        for cand, ref, _ in PAIRS:
            self.assertIn(cand, known)
            self.assertIn(ref, known)


if __name__ == "__main__":
    unittest.main()
