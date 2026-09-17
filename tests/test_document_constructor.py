import json
import shutil
import unittest
from pathlib import Path

from optima.rag.document_constructor import OptimaDocumentConstructor, _stringify


def _base_json(functions):
    return {
        "project": {"name": "test", "root": "/tmp/test", "language": "cpp"},
        "files": [{"path": "file.cpp", "name": "file.cpp", "relative_path": "file.cpp",
                   "functions": functions}],
    }


class StringifyTests(unittest.TestCase):
    def test_none_becomes_empty_string(self):
        self.assertEqual(_stringify(None), "")

    def test_plain_string_passes_through(self):
        self.assertEqual(_stringify("hello"), "hello")

    def test_number_and_bool(self):
        self.assertEqual(_stringify(3), "3")
        self.assertEqual(_stringify(True), "True")

    def test_dict_renders_as_key_value_text(self):
        self.assertEqual(_stringify({"name": "board", "type": "Board*"}), "name: board, type: Board*")

    def test_list_of_dicts_renders_each_item(self):
        value = [{"name": "board", "type": "Board*"}, {"name": "x", "type": "int"}]
        self.assertEqual(_stringify(value), "name: board, type: Board*; name: x, type: int")

    def test_list_of_strings_still_joins_with_comma_semantics(self):
        self.assertEqual(_stringify(["a", "b"]), "a; b")


class DocumentConstructorTests(unittest.TestCase):
    root = Path(__file__).parent / ".document_constructor_artifacts"

    def setUp(self):
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write(self, functions) -> Path:
        path = self.root / "enhanced_m1.json"
        path.write_text(json.dumps(_base_json(functions)), encoding="utf-8")
        return path

    def test_dict_valued_enrichment_fields_do_not_drop_the_function(self):
        # The exact shape requirement 7 calls out: "inputs": [{"name":
        # "board", "type": "Board*"}] -- structured, not a flat string list.
        functions = [{
            "id": "file.cpp::f::1", "name": "f", "qualified_name": "f",
            "source_code": "void f(Board* board) {}",
            "enrichment": {
                "status": "completed",
                "purpose": "Does something with the board.",
                "behavior": "Mutates board state.",
                "inputs": [{"name": "board", "type": "Board*"}],
                "outputs": [{"name": "result", "type": "int"}],
                "complexity": {"time": "O(1)", "space": "O(1)"},
            },
        }]
        path = self._write(functions)
        docs = OptimaDocumentConstructor(representation_mode="hybrid").construct_documents_from_json(
            path, enrichment_model="m1"
        )
        self.assertEqual(len(docs), 1)
        self.assertIn("board", docs[0].page_content)
        self.assertIn("Board*", docs[0].page_content)

    def test_unrecovered_enrichment_still_produces_a_document(self):
        # No "enrichment" content (json_parse_status == "failed" case) --
        # the function must still become a document, built from base fields.
        functions = [{
            "id": "file.cpp::f::2", "name": "g", "qualified_name": "g",
            "source_code": "int g() { return 2; }",
            "enrichment": {"status": "failed", "json_parse_status": "failed"},
        }]
        path = self._write(functions)
        docs = OptimaDocumentConstructor(representation_mode="hybrid").construct_documents_from_json(
            path, enrichment_model="m1"
        )
        self.assertEqual(len(docs), 1)
        self.assertIn("g", docs[0].page_content)

    def test_document_count_matches_base_function_count(self):
        functions = [
            {"id": f"file.cpp::f{i}::{i}", "name": f"f{i}", "qualified_name": f"f{i}",
             "source_code": f"int f{i}() {{ return {i}; }}",
             "enrichment": {"status": "completed", "purpose": "p", "inputs": [{"name": "x", "type": "int"}]}}
            for i in range(6)
        ]
        path = self._write(functions)
        docs = OptimaDocumentConstructor(representation_mode="hybrid").construct_documents_from_json(
            path, enrichment_model="m1"
        )
        self.assertEqual(len(docs), 6)


if __name__ == "__main__":
    unittest.main()
