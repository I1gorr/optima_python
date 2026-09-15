import json
import shutil
import unittest
from pathlib import Path

from optima.rag.embedding_simple import (
    embed_corpus,
    embedding_alias,
    index_directory,
    load_index_metadata,
    register_embedding_model,
    resolve_embedding_model,
)
from optima.rag.evaluation import weighted_overall_score


class EmbeddingRegistryTests(unittest.TestCase):
    root = Path(__file__).parent / ".embedding_registry_artifacts"

    @classmethod
    def setUpClass(cls):
        if cls.root.exists():
            shutil.rmtree(cls.root)
        cls.root.mkdir(parents=True)
        cls.json_path = cls.root / "base.json"
        cls.json_path.write_text(json.dumps({
            "project": {"name": "test"},
            "files": [{"path": "x.cpp", "functions": [{
                "id": "x.cpp::f::1", "name": "f", "qualified_name": "f",
                "source_code": "int f() { return 1; }"
            }]}]
        }), encoding="utf-8")
        register_embedding_model("test-mock", "mock")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_alias_resolution_and_canonical_path(self):
        self.assertEqual(embedding_alias("BAAI/bge-small-en-v1.5"), "bge-small")
        self.assertEqual(resolve_embedding_model("test-mock").model_name, "mock")
        self.assertEqual(
            index_directory(self.root, "raw", "test-mock"),
            self.root / "indexes" / "raw" / "test-mock",
        )

    def test_metadata_and_skip_without_force(self):
        first = embed_corpus(self.json_path, None, "test-mock", self.root)
        self.assertTrue(first["success"])
        metadata = load_index_metadata(index_directory(self.root, "raw", "test-mock"))
        self.assertEqual(metadata["embedding_alias"], "test-mock")
        self.assertEqual(metadata["corpus"], "raw")
        second = embed_corpus(self.json_path, None, "test-mock", self.root)
        self.assertTrue(second["skipped"])

    def test_weighted_score(self):
        self.assertAlmostEqual(
            weighted_overall_score({
                "mrr": 1, "recall_at_5": 0.8,
                "hit_rate_at_5": 0.6, "precision_at_5": 0.4,
            }),
            0.76,
        )


if __name__ == "__main__":
    unittest.main()
