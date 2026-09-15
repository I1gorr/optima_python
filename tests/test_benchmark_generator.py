import unittest

from optima.rag.benchmark_generator import (
    BenchmarkQueryGenerator,
    validate_benchmark_queries,
)


def function(fid, name, summary, calls=None):
    return {
        "id": fid,
        "name": name,
        "qualified_name": name,
        "return_type": "int",
        "parameters": [{"name": "input_value", "type": "int"}],
        "source_code": "return input_value;",
        "calls": calls or [],
        "enrichment": {
            "summary": summary,
            "purpose": summary,
            "behavior": summary,
        },
    }


class BenchmarkGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.functions = [
            function("a", "parseRecord", "Converts encoded input into structured fields", [{"id": "b", "name": "storeFields"}]),
            function("b", "storeFields", "Persists structured fields in the lookup state", [{"id": "c", "name": "emitResult"}]),
            function("c", "emitResult", "Produces a compact result for the caller"),
            function("d", "validateInput", "Checks input constraints before processing"),
        ]

    def test_default_distribution_is_twenty_five_thirty_five_forty(self):
        queries = BenchmarkQueryGenerator(1).generate_benchmark_queries(self.functions, 20)
        counts = {level: sum(q["difficulty"] == level for q in queries) for level in ("easy", "moderate", "difficult")}
        self.assertEqual(counts, {"easy": 5, "moderate": 7, "difficult": 8})

    def test_queries_do_not_leak_target_identifiers_and_are_unique(self):
        queries = BenchmarkQueryGenerator(2).generate_benchmark_queries(self.functions, 20)
        names = {"parseRecord", "storeFields", "emitResult", "validateInput"}
        for item in queries:
            self.assertFalse(names & set(item["query"].split()))
            self.assertTrue(item["is_safe"])
            self.assertIn("reasoning_depth", item)
            self.assertIn("directness_score", item)
            self.assertIn("query_quality_score", item)
        self.assertEqual(len({item["query"].lower() for item in queries}), len(queries))

    def test_graph_evidence_supports_multi_hop_relevance(self):
        queries = BenchmarkQueryGenerator(3).generate_benchmark_queries(self.functions, 20)
        multi_hop = [q for q in queries if q["category"] == "multi_hop"]
        self.assertTrue(multi_hop)
        self.assertTrue(any(len(q["relevant_functions"]) >= 3 for q in multi_hop))

    def test_validation_reports_duplicates_and_identifier_leaks(self):
        queries = [{
            "query_id": "q1",
            "query": "Where does parseRecord run?",
            "relevant_functions": ["a"],
            "difficulty": "easy",
            "category": "functionality",
        }, {
            "query_id": "q2",
            "query": "Where does parseRecord run?",
            "relevant_functions": ["a"],
            "difficulty": "easy",
            "category": "functionality",
        }]
        report = validate_benchmark_queries(queries, self.functions)
        self.assertEqual(report["duplicate_queries"], 1)
        self.assertEqual(report["identifier_leaks"], 2)
        self.assertEqual(report["unsafe_queries"], 2)

    def test_identifier_lookup_queries_are_rejected(self):
        generator = BenchmarkQueryGenerator()
        safe, details = generator._is_query_safe(
            "Which function handles parseRecord?",
            self.functions[0],
        )
        self.assertFalse(safe)
        self.assertTrue(details["contains_target_identifier"])

    def test_difficult_queries_have_low_directness(self):
        queries = BenchmarkQueryGenerator(4).generate_benchmark_queries(self.functions, 20)
        difficult = [item for item in queries if item["difficulty"] == "difficult"]
        self.assertTrue(difficult)
        self.assertTrue(all(item["directness_score"] <= 0.4 for item in difficult))


if __name__ == "__main__":
    unittest.main()
