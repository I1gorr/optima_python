import unittest

from langchain_core.documents import Document

from optima.rag.evaluation import OptimaRetrievalEvaluator


class _Retriever:
    def __init__(self, documents):
        self.documents = documents

    def get_relevant_documents(self, _query):
        return self.documents


class _VectorStore:
    def __init__(self, function_ids):
        self.documents = [
            Document(
                page_content=f"function {function_id}",
                metadata={"function_id": function_id},
            )
            for function_id in function_ids
        ]

    def as_retriever(self, search_kwargs=None):
        return _Retriever(self.documents[: search_kwargs.get("k", 4)])


def query(relevant_functions):
    return {
        "query_id": "q_test",
        "query": "where is the implementation?",
        "relevant_functions": relevant_functions,
    }


class RetrievalMetricsTests(unittest.TestCase):
    def test_mrr_uses_first_relevant_rank(self):
        evaluator = OptimaRetrievalEvaluator()
        cases = [
            (["a"], 1.0),
            (["x", "a"], 0.5),
            (["x", "y", "z", "w", "a"], 0.2),
            (["x", "y", "z", "w", "v", "u", "t", "s", "r", "a"], 0.1),
            (["x", "y", "z"], 0.0),
        ]

        for function_ids, expected in cases:
            with self.subTest(function_ids=function_ids):
                result = evaluator.evaluate_retrieval(
                    _VectorStore(function_ids), [query(["a"])], k=10
                )
                self.assertAlmostEqual(result["mrr"], expected)

    def test_mrr_mixed_ranks_and_multiple_relevant_documents(self):
        evaluator = OptimaRetrievalEvaluator()
        queries = [
            query(["a"]),  # rank 1
            query(["b"]),  # rank 2
            query(["c"]),  # rank 5
            query(["d"]),  # no hit in top 5
            query(["b", "c"]),  # first relevant is rank 2
        ]
        stores = [
            _VectorStore(["a"]),
            _VectorStore(["x", "b"]),
            _VectorStore(["x", "y", "z", "w", "c"]),
            _VectorStore(["x", "y", "z", "w", "v"]),
            _VectorStore(["x", "b", "c"]),
        ]
        scores = [
            evaluator.evaluate_retrieval(store, [item], k=5)["mrr"]
            for store, item in zip(stores, queries)
        ]
        self.assertEqual(scores, [1.0, 0.5, 0.2, 0.0, 0.5])
        self.assertAlmostEqual(sum(scores) / len(scores), 0.44)

    def test_recall_and_hit_use_their_requested_cutoff(self):
        evaluator = OptimaRetrievalEvaluator()
        result = evaluator.evaluate_retrieval(
            _VectorStore(["wrong", "target"]), [query(["target"])], k=10
        )
        self.assertEqual(result["recall_at_1"], 0.0)
        self.assertEqual(result["recall_at_3"], 1.0)
        self.assertEqual(result["hit_rate_at_1"], 0.0)
        self.assertEqual(result["hit_rate_at_5"], 1.0)


if __name__ == "__main__":
    unittest.main()
