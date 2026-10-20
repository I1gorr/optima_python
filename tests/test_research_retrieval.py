import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from optima.rag.research.config import PipelineConfig, baseline_rag, compiler_hybrid_rag
from optima.rag.research.corpus import load_corpus
from optima.rag.research.dense import DenseIndex, EmbeddingCache, HashingEncoder
from optima.rag.research.fusion import RankedList, rrf_fuse
from optima.rag.research.graph import GraphIndex
from optima.rag.research.lexical import BM25Index, ExactSymbolMatcher, normalize_query, tokenize
from optima.rag.research.pipeline import ResearchPipeline
from optima.rag.research.rerank import CrossEncoderReranker, RerankerError, RerankerUnavailable

from research_fixtures import make_id, write_fixture


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.paths = write_fixture(Path(self._tmp.name))
        self.corpus = load_corpus(self.paths["base"], self.paths["enriched"])

    def config(self, name="ablation", **sections):
        data = {"name": name, "base_json": str(self.paths["base"]),
                "enriched_json": str(self.paths["enriched"]), "embedding_model": "hashing",
                "retrieval": {"representations": ["raw"]}}
        data.update(sections)
        return PipelineConfig.from_dict(data)

    def pipeline(self, config, **kwargs):
        return ResearchPipeline(config, corpus=self.corpus, encoder=HashingEncoder(), **kwargs)


class LexicalTests(unittest.TestCase):
    def test_tokenizer_splits_identifiers_but_keeps_whole_token(self):
        tokens = tokenize("cJSON_ParseWithOpts parse_json")
        for expected in ("cjson_parsewithopts", "cjson", "parse", "with", "opts", "parse_json", "json"):
            self.assertIn(expected, tokens)

    def test_query_normalisation_is_shared_and_handles_none(self):
        self.assertEqual(normalize_query("  a \n\t b  "), "a b")
        self.assertEqual(normalize_query(None), "")

    def test_bm25_ranks_exact_terms_and_is_deterministic(self):
        index = BM25Index().build(["a", "b", "c"], ["alpha beta", "beta gamma gamma", "delta"])
        first = index.search("gamma", 5)
        self.assertEqual([i for i, _ in first], ["b"])
        self.assertEqual(index.search("beta", 5), index.search("beta", 5))
        self.assertEqual([i for i, _ in index.search("beta", 5)], ["a", "b"])  # tie -> index order... a shorter
        self.assertEqual(index.search("", 5), [])
        self.assertEqual(index.search("zzz", 5), [])
        self.assertEqual(BM25Index().build([], []).search("x", 3), [])

    def test_bm25_idf_prefers_rare_terms(self):
        index = BM25Index().build(["a", "b", "c"], ["common rare", "common", "common"])
        self.assertEqual(index.search("common rare", 3)[0][0], "a")

    def test_exact_matcher_only_treats_symbol_shaped_tokens_as_symbols(self):
        matcher = ExactSymbolMatcher([("1", "parse_json", "parse_json"), ("2", "add", "add"),
                                      ("3", "read_token", "ns::read_token")])
        self.assertEqual(matcher.search("where is parse_json defined", 5)[0][0], "1")
        self.assertEqual(matcher.search("please add the numbers", 5), [])   # plain word, not a symbol
        self.assertEqual(matcher.search("see `add` here", 5)[0][0], "2")   # quoted -> symbol
        self.assertEqual(matcher.search("call ns::read_token", 5)[0], ("3", 3.0))


class FusionTests(unittest.TestCase):
    def test_rrf_scores_and_order(self):
        a = RankedList("a", [("x", 9.0), ("y", 5.0), ("z", 1.0)])
        b = RankedList("b", [("y", 0.9), ("w", 0.5)])
        fused = rrf_fuse([a, b], rrf_k=60)
        self.assertEqual([f.id for f in fused], ["y", "x", "w", "z"])
        self.assertAlmostEqual(fused[0].score, 1 / 62 + 1 / 61)
        # provenance keeps each source's rank and native score
        y = fused[0]
        self.assertEqual([(c.source, c.rank, c.score) for c in y.contributions],
                         [("a", 2, 5.0), ("b", 1, 0.9)])

    def test_rrf_dedupes_within_a_list_weights_and_limit(self):
        a = RankedList("a", [("x", 1.0), ("x", 0.9), ("y", 0.5)])
        fused = rrf_fuse([a], rrf_k=10)
        self.assertEqual([f.id for f in fused], ["x", "y"])
        self.assertEqual(fused[1].contributions[0].rank, 2)       # duplicate did not consume a rank
        weighted = rrf_fuse([RankedList("a", [("x", 1)], 1.0), RankedList("b", [("y", 1)], 3.0)], 60)
        self.assertEqual(weighted[0].id, "y")
        self.assertEqual(len(rrf_fuse([a], 10, limit=1)), 1)
        self.assertEqual(rrf_fuse([], 60), [])
        with self.assertRaises(ValueError):
            rrf_fuse([a], 0)

    def test_rrf_ties_break_deterministically(self):
        a = RankedList("a", [("p", 1)])
        b = RankedList("b", [("q", 1)])
        self.assertEqual([f.id for f in rrf_fuse([a, b])], ["p", "q"])
        self.assertEqual([f.id for f in rrf_fuse([b, a])], ["q", "p"])


class DenseTests(FixtureCase):
    def test_dense_search_ranks_semantically_similar_text_first(self):
        encoder = HashingEncoder()
        ids = ["a", "b"]
        index = DenseIndex.build(ids, ["parse json document", "format floating number"], encoder)
        top = index.search(encoder.encode_queries(["json parse"])[0], 2)
        self.assertEqual(top[0][0], "a")
        self.assertGreater(top[0][1], top[1][1])

    def test_embedding_cache_hit_returns_identical_vectors(self):
        encoder = HashingEncoder()
        cache = EmbeddingCache(Path(self._tmp.name) / "cache")
        first = DenseIndex.build(["a", "b"], ["one two", "three"], encoder, cache)
        second = DenseIndex.build(["a", "b"], ["one two", "three"], encoder, cache)
        self.assertFalse(first.build_info["cache_hit"])
        self.assertTrue(second.build_info["cache_hit"])
        np.testing.assert_array_equal(first.matrix, second.matrix)
        changed = DenseIndex.build(["a", "b"], ["one two", "four"], encoder, cache)
        self.assertFalse(changed.build_info["cache_hit"])

    def test_unavailable_model_raises_instead_of_falling_back_to_mock(self):
        from optima.rag.research.dense import EmbeddingUnavailable, SentenceTransformerEncoder
        with mock.patch("sentence_transformers.SentenceTransformer", side_effect=OSError("no such model")):
            with self.assertRaisesRegex(EmbeddingUnavailable, "could not load embedding model"):
                SentenceTransformerEncoder("bge-small")


class GraphTests(FixtureCase):
    def setUp(self):
        super().setUp()
        self.graph = GraphIndex(self.corpus)

    def test_edges_come_only_from_resolved_data(self):
        self.assertEqual(self.graph.neighbours(make_id("parse"), "callees"),
                         [make_id("read"), make_id("alloc")])
        self.assertEqual(self.graph.neighbours(make_id("read"), "callers"), [make_id("parse")])
        self.assertEqual(self.graph.stats["call_edges"], 4)       # malloc is unresolved: no node
        self.assertEqual(self.graph.stats["dependency_edges"], 0)  # `dependencies` empty in data
        # LLM dependency names become edges only on an exact, unique qualified-name match:
        # "read_token" -> the one function whose qualified name is exactly that; "std::string" -> none
        self.assertEqual(self.graph.stats["enriched_dependency_edges"], 1)
        self.assertEqual(self.graph.neighbours(make_id("parse"), "enriched_dependencies"), [make_id("read")])
        self.assertEqual(self.graph.stats["enriched_dependency_names_unmatched"], 1)

    def test_expansion_depth_and_provenance(self):
        found = {c.id: c for c in self.graph.expand([(make_id("parse"), 1)], ["callees"], max_depth=2)}
        self.assertEqual(set(found), {make_id("read"), make_id("alloc"), make_id("digit")})
        self.assertEqual(found[make_id("read")].hop, 1)
        self.assertEqual(found[make_id("digit")].hop, 2)
        evidence = found[make_id("digit")].evidence[0]
        self.assertEqual((evidence.seed, evidence.parent, evidence.relation, evidence.hop),
                         (make_id("parse"), make_id("read"), "callees", 2))
        self.assertIn("calls", evidence.edge_sources)
        depth_one = {c.id for c in self.graph.expand([(make_id("parse"), 1)], ["callees"], max_depth=1)}
        self.assertNotIn(make_id("digit"), depth_one)

    def test_direction_selection(self):
        callers = {c.id for c in self.graph.expand([(make_id("digit"), 1)], ["callers"], max_depth=2)}
        self.assertEqual(callers, {make_id("read"), make_id("parse")})
        self.assertEqual(self.graph.expand([(make_id("digit"), 1)], ["callees"]), [])

    def test_limits_bound_the_traversal(self):
        capped = self.graph.expand([(make_id("parse"), 1)], ["callees"], max_depth=2, max_added=1)
        self.assertEqual(len([c for c in capped if not c.is_seed]), 1)
        narrow = self.graph.expand([(make_id("parse"), 1)], ["callees"], max_neighbors=1)
        self.assertEqual([c.id for c in narrow], [make_id("read")])
        self.assertEqual(self.graph.expand([(make_id("parse"), 1)], ["callees"], max_depth=0), [])
        with self.assertRaises(ValueError):
            self.graph.expand([(make_id("parse"), 1)], ["bogus"])

    def test_seed_reached_from_another_seed_is_corroboration_not_added_candidate(self):
        found = self.graph.expand([(make_id("parse"), 1), (make_id("read"), 2)], ["callees"])
        by_id = {c.id: c for c in found}
        self.assertTrue(by_id[make_id("read")].is_seed)
        self.assertEqual(by_id[make_id("read")].evidence[0].seed, make_id("parse"))

    def test_unknown_seed_ids_are_ignored(self):
        self.assertEqual(self.graph.expand([("nope", 1)], ["callees"]), [])

    def test_graph_ordering_inherits_seed_rank(self):
        found = self.graph.expand([(make_id("print"), 1), (make_id("parse"), 2)], ["callees"])
        ordered = [c.id for c in self.graph.order_for_fusion(found)]
        self.assertEqual(ordered[0], make_id("format"))            # neighbour of the rank-1 seed


class RerankTests(unittest.TestCase):
    def test_reorders_by_score_preserving_ties_and_scores(self):
        scores = {"a": 0.1, "b": 0.9, "c": 0.9}
        rr = CrossEncoderReranker("ms-marco-minilm",
                                  predict_fn=lambda pairs: [scores[d] for _, d in pairs])
        ranked = rr.rerank("q", [("a", "a"), ("b", "b"), ("c", "c")])
        self.assertEqual(ranked, [("b", 0.9), ("c", 0.9), ("a", 0.1)])

    def test_inference_error_is_wrapped(self):
        def boom(_pairs):
            raise RuntimeError("cuda oom")
        rr = CrossEncoderReranker("ms-marco-minilm", predict_fn=boom)
        with self.assertRaisesRegex(RerankerError, "cuda oom"):
            rr.rerank("q", [("a", "a")])
        wrong = CrossEncoderReranker("ms-marco-minilm", predict_fn=lambda p: [1.0])
        with self.assertRaises(RerankerError):
            wrong.rerank("q", [("a", "a"), ("b", "b")])

    def test_load_failure_raises_unavailable_and_is_not_silently_replaced(self):
        rr = CrossEncoderReranker("some/nonexistent-model", device="cpu")
        with mock.patch("sentence_transformers.CrossEncoder", side_effect=OSError("offline")):
            with self.assertRaisesRegex(RerankerUnavailable, "could not load cross-encoder"):
                rr.load()

    def test_model_is_loaded_once_per_process(self):
        created = []

        class Fake:
            def __init__(self, *a, **k):
                created.append(1)
        with mock.patch("sentence_transformers.CrossEncoder", Fake):
            import optima.rag.research.rerank as rerank_module
            rerank_module._LOADED.clear()
            CrossEncoderReranker("fake/ce", device="cpu").load()
            CrossEncoderReranker("fake/ce", device="cpu").load()
        self.assertEqual(len(created), 1)

    def test_empty_candidates(self):
        rr = CrossEncoderReranker("ms-marco-minilm", predict_fn=lambda p: [])
        self.assertEqual(rr.rerank("q", []), [])


class PipelineTests(FixtureCase):
    def test_dense_only_bm25_only_and_hybrid_all_work(self):
        query = "parse json tokens"
        results = {}
        for name, kw in {"dense": dict(dense=True, bm25=False), "bm25": dict(dense=False, bm25=True),
                         "hybrid": dict(dense=True, bm25=True)}.items():
            pipe = self.pipeline(self.config(retrieval={"representations": ["raw"], **kw}))
            results[name] = pipe.search(query)
            self.assertEqual(results[name].results[0].function_id, make_id("parse"), name)
        methods = {m for it in results["hybrid"].results for m in it.methods}
        self.assertEqual(methods, {"dense", "bm25"})
        self.assertEqual({m for it in results["bm25"].results for m in it.methods}, {"bm25"})

    def test_results_are_deduplicated_ranked_and_carry_provenance(self):
        pipe = self.pipeline(self.config(retrieval={"representations": ["raw", "semantic", "compiler"],
                                                    "bm25": True}))
        result = pipe.search("allocate a node")
        ids = [it.function_id for it in result.results]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual([it.rank for it in result.results], list(range(1, len(ids) + 1)))
        top = result.results[0]
        self.assertEqual(top.function_id, make_id("alloc"))
        self.assertTrue({"dense:raw", "bm25:semantic"} <= {s["source"] for s in top.sources})
        self.assertEqual(top.representations, ["compiler", "raw", "semantic"])

    def test_multi_representation_fusion_records_which_representation_found_what(self):
        # "Purpose" text exists only in the semantic representation
        pipe = self.pipeline(self.config(retrieval={"representations": ["raw", "semantic"],
                                                    "dense": False, "bm25": True}))
        result = pipe.search("decimal digit predicate")
        top = next(it for it in result.results if it.function_id == make_id("digit"))
        self.assertIn("semantic", top.representations)
        only_raw = self.pipeline(self.config(retrieval={"representations": ["raw"], "dense": False,
                                                        "bm25": True})).search("decimal digit predicate")
        self.assertTrue(all(it.representations == ["raw"] for it in only_raw.results))

    def test_missing_enrichment_function_remains_retrievable(self):
        pipe = self.pipeline(self.config(retrieval={"representations": ["semantic"], "dense": False,
                                                    "bm25": True}))
        result = pipe.search("hash_string")
        self.assertEqual(result.results[0].function_id, make_id("hash"))  # failed enrichment, kept by name

    def test_empty_and_whitespace_queries_are_reported_not_crashed(self):
        pipe = self.pipeline(self.config())
        for query in ("", "   ", None):
            result = pipe.search(query)
            self.assertEqual(result.error, "empty_query")
            self.assertEqual(result.results, [])

    def test_graph_expansion_adds_candidates_with_evidence_and_can_be_disabled(self):
        base_kwargs = dict(retrieval={"representations": ["raw"], "dense": False, "bm25": True})
        off = self.pipeline(self.config(graph={"enabled": False}, **base_kwargs)).search("parse_json")
        on = self.pipeline(self.config(graph={"enabled": True, "relations": ["callees"], "seed_k": 1},
                                       **base_kwargs)).search("parse_json")
        off_ids = [it.function_id for it in off.results]
        on_ids = [it.function_id for it in on.results]
        self.assertEqual(off.counts["graph_added"], 0)
        self.assertTrue(all(not it.graph_evidence for it in off.results))
        self.assertGreater(on.counts["candidates_after_graph"], on.counts["first_stage_candidates"] - 1)
        graph_items = [it for it in on.results if it.graph_evidence]
        self.assertTrue(graph_items)
        self.assertTrue(all("graph" in it.methods for it in graph_items))
        self.assertEqual(off_ids[0], on_ids[0])
        # disabling the graph leaves first-stage rank/score intact on the graph run
        for it in on.results:
            if it.first_stage_rank is not None:
                self.assertIn(it.function_id, off.pool_before_graph)

    def test_graph_only_candidates_appear_even_when_first_stage_missed_them(self):
        # query matches only parse_json lexically; its callee is_digit's callee... via depth 2
        pipe = self.pipeline(self.config(
            retrieval={"representations": ["raw"], "dense": False, "bm25": True, "candidate_k": 1},
            graph={"enabled": True, "relations": ["callees"], "max_depth": 2, "seed_k": 1}, top_k=5))
        result = pipe.search("parse_json")
        ids = {it.function_id: it for it in result.results}
        self.assertIn(make_id("digit"), ids)
        self.assertIsNone(ids[make_id("digit")].first_stage_rank)
        self.assertEqual(ids[make_id("digit")].graph_evidence[0]["hop"], 2)

    def test_rerank_is_optional_records_both_ranks_and_survives_errors(self):
        scores = {make_id("read"): 10.0}
        predict = lambda pairs: [scores.get(self._id_of(doc), 0.0) for _, doc in pairs]  # noqa: E731
        rr = CrossEncoderReranker("ms-marco-minilm", predict_fn=predict)
        cfg = self.config(retrieval={"representations": ["raw"], "dense": False, "bm25": True,
                                     "exact_match": False},
                          rerank={"enabled": True, "pool_size": 8, "text_representations": ["raw"]}, top_k=5)
        pipe = self.pipeline(cfg, reranker=rr)
        result = pipe.search("is_digit")
        self.assertTrue(result.rerank_applied)
        top = result.results[0]
        self.assertEqual(top.function_id, make_id("read"))
        self.assertEqual(top.score_type, "cross_encoder")
        self.assertEqual(top.rerank_score, 10.0)
        self.assertEqual(top.pre_rerank_rank, 2)           # BM25 had the definition first
        self.assertEqual(top.first_stage_rank, 2)
        self.assertIsNotNone(top.first_stage_rank)
        # an inference failure keeps the original ranking and flags the query
        broken = CrossEncoderReranker("ms-marco-minilm",
                                      predict_fn=lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
        result2 = self.pipeline(cfg, reranker=broken).search("is_digit")
        self.assertFalse(result2.rerank_applied)
        self.assertIn("boom", result2.rerank_error)
        self.assertEqual(result2.results[0].score_type, "rrf")
        # and with rerank disabled nothing rerank-related is present
        plain = self.pipeline(self.config(retrieval={"representations": ["raw"], "dense": False,
                                                     "bm25": True})).search("is_digit")
        self.assertIsNone(plain.results[0].rerank_score)
        self.assertEqual([it.function_id for it in plain.results],
                         [it.function_id for it in result2.results])

    def _id_of(self, doc):
        for ident, record in self.corpus.records.items():
            if record.source_code and record.source_code in doc:
                return ident
        return None

    def test_pipeline_ranking_is_deterministic(self):
        cfg = compiler_hybrid_rag(str(self.paths["base"]), str(self.paths["enriched"]),
                                  embedding_model="hashing")
        a = ResearchPipeline(cfg, corpus=self.corpus, encoder=HashingEncoder()).search("parse tokens")
        b = ResearchPipeline(cfg, corpus=self.corpus, encoder=HashingEncoder()).search("parse tokens")
        self.assertEqual([(i.function_id, i.score) for i in a.results],
                         [(i.function_id, i.score) for i in b.results])


if __name__ == "__main__":
    unittest.main()
