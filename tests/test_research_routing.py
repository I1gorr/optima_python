import json
import tempfile
import unittest
from pathlib import Path

from optima.rag.research.benchmark import BenchmarkError, build_developer_benchmark
from optima.rag.research.config import ConfigError, PipelineConfig
from optima.rag.research.dense import HashingEncoder
from optima.rag.research.pipeline import ResearchPipeline
from optima.rag.research.routing import CLASSES, classify_query, profile_for
from research_fixtures import write_fixture
from optima.rag.research.corpus import load_corpus


def make_corpus():
    tmp = tempfile.mkdtemp()
    paths = write_fixture(Path(tmp))
    return load_corpus(paths['base'], paths['enriched'])


class ClassifierTests(unittest.TestCase):
    def test_literal_symbol_structural_natural(self):
        self.assertEqual(classify_query("where does it print 'Failed to open the file.'"), "literal")
        self.assertEqual(classify_query("who prints \"Wrong parameters\""), "literal")
        self.assertEqual(classify_query("Which functions call cJSON_Delete?"), "structural")
        self.assertEqual(classify_query("what breaks if I change getRookAttacks"), "structural")
        self.assertEqual(classify_query("Where is the function resetGameVariables defined?"), "symbol")
        self.assertEqual(classify_query("how do I parse a json string"), "natural")
        self.assertEqual(classify_query(""), "natural")

    def test_identifier_search_is_not_an_impact_question(self):
        self.assertEqual(classify_query("Which functions use the identifier keySize?"), "symbol")

    def test_single_quoted_word_is_not_a_literal(self):
        self.assertEqual(classify_query("what does 'foo' do"), "symbol")     # a quoted token is a symbol, not a message

    def test_depends_on_requires_a_symbol(self):
        self.assertEqual(classify_query("what depends on the parser"), "natural")
        self.assertEqual(classify_query("what depends on parse_object"), "structural")

    def test_classifier_ignores_case_and_whitespace_noise(self):
        self.assertEqual(classify_query("  WHICH   functions CALL   cJSON_Delete ?  "), "structural")

    def test_profiles_default_to_neutral_and_overrides_win(self):
        self.assertEqual(profile_for("natural", None), {})
        self.assertEqual(profile_for("literal", {"literal": {"bm25_weight": 3}}), {"bm25_weight": 3})
        self.assertEqual(set(CLASSES), {"literal", "structural", "symbol", "natural"})


class RoutingConfigTests(unittest.TestCase):
    def base(self, **extra):
        data = {"name": "t", "embedding_model": "hashing", "retrieval": {"representations": ["raw"], "dense": True, "bm25": True}}
        data.update(extra)
        return data

    def test_hash_unchanged_when_routing_disabled_and_changes_when_enabled(self):
        plain = PipelineConfig.from_dict(self.base())
        disabled = PipelineConfig.from_dict(self.base(routing={"enabled": False}))
        enabled = PipelineConfig.from_dict(self.base(routing={"enabled": True, "profiles": {"literal": {"bm25_weight": 3}}}))
        self.assertEqual(plain.config_hash(), disabled.config_hash())
        self.assertNotEqual(plain.config_hash(), enabled.config_hash())

    def test_invalid_profiles_are_rejected(self):
        with self.assertRaises(ConfigError):
            PipelineConfig.from_dict(self.base(routing={"enabled": True, "profiles": {"nonsense": {}}}))
        with self.assertRaises(ConfigError):
            PipelineConfig.from_dict(self.base(routing={"enabled": True, "profiles": {"literal": {"bogus": 1}}}))
        with self.assertRaises(ConfigError):
            PipelineConfig.from_dict(self.base(routing={"enabled": True, "profiles": {"literal": {"bm25_weight": -1}}}))

    def test_roundtrip(self):
        cfg = PipelineConfig.from_dict(self.base(routing={"enabled": True, "profiles": {"natural": {"representation_weights": {"raw": 2}}}}))
        again = PipelineConfig.from_dict(cfg.to_dict())
        self.assertEqual(cfg.config_hash(), again.config_hash())


class RoutedPipelineTests(unittest.TestCase):
    def build(self, routing):
        corpus = make_corpus()
        cfg = PipelineConfig.from_dict({"name": "t", "embedding_model": "hashing",
                                        "retrieval": {"representations": ["raw"], "dense": True, "bm25": True, "exact_match": True},
                                        "routing": routing})
        pipe = ResearchPipeline(cfg, corpus=corpus, encoder=HashingEncoder())
        pipe.build()
        return pipe, corpus

    def test_neutral_routing_matches_unrouted_ranking(self):
        routed, corpus = self.build({"enabled": True})
        plain, _ = self.build({"enabled": False})
        name = next(iter(corpus.records.values())).name
        for q in (f"where is {name} defined", "how do I add two numbers", "who calls " + name):
            a = [r.function_id for r in routed.search(q).results]
            b = [r.function_id for r in plain.search(q).results]
            self.assertEqual(a, b)

    def test_route_is_recorded_and_weights_take_effect(self):
        pipe, corpus = self.build({"enabled": True, "profiles": {"natural": {"bm25_weight": 0.0, "exact_weight": 0.0}}})
        res = pipe.search("how do I add two numbers")
        self.assertEqual(res.counts["route"], "natural")
        sources = {c["source"] for r in res.results for c in r.sources}
        self.assertTrue(sources)
        self.assertTrue(all(not s.startswith(("bm25", "exact")) for s in sources), sources)


class DeveloperBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.corpus = make_corpus()
        self.records = list(self.corpus.records.values())

    def authored(self, *entries):
        return {"queries": list(entries)}

    def test_all_three_label_kinds_resolve(self):
        r = self.records[0]
        lit = "a distinctive message"
        r.source_code = r.source_code + f' puts("{lit}");'
        bench = build_developer_benchmark(self.corpus, self.authored(
            {"intent": "task", "query": "how do I do x", "targets": [{"file": "", "qualified_name": r.qualified_name, "grade": 2}]},
            {"intent": "error_message", "query": f"where is '{lit}'", "string_literal": lit},
            ), benchmark_id="t")
        self.assertEqual(bench["counts"], {"task": 1, "error_message": 1})
        by = {q["category"]: q for q in bench["queries"]}
        self.assertEqual(by["error_message"]["label_source"], "source_string_literal")
        self.assertIn(r.id, by["error_message"]["relevance"])
        self.assertEqual(by["task"]["relevance"][r.id], 2.0)

    def test_unknown_intent_and_ambiguous_literal_fail_loudly(self):
        with self.assertRaises(BenchmarkError):
            build_developer_benchmark(self.corpus, self.authored({"intent": "other", "query": "x", "targets": []}))
        with self.assertRaises(BenchmarkError):
            build_developer_benchmark(self.corpus, self.authored({"intent": "error_message", "query": "x", "string_literal": "zzzz-not-present"}))
        with self.assertRaises(BenchmarkError):
            build_developer_benchmark(self.corpus, self.authored({"intent": "task", "query": "x", "targets": [{"file": "nope.c", "qualified_name": "nothing"}]}))

    def test_literal_matching_too_many_functions_is_rejected(self):
        for r in self.records:
            r.source_code += ' common_literal_text '
        with self.assertRaises(BenchmarkError):
            build_developer_benchmark(self.corpus, self.authored(
                {"intent": "error_message", "query": "q", "string_literal": "common_literal_text", "max_matches": 1}))


if __name__ == "__main__":
    unittest.main()
