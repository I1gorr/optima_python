"""
Document constructor for Optima RAG system.
Handles conversion of raw and enriched JSON files to LangChain Documents.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _stringify(value: Any) -> str:
    """Render any JSON-shaped value (str/number/bool/dict/list/None) as
    plain text for an embedding document, instead of assuming every
    enrichment field is a flat string or a list of strings. An unrestricted
    LLM may legitimately return structured values -- e.g.
    ``"inputs": [{"name": "board", "type": "Board*"}]`` -- and that must
    still become readable text, never a rejected/dropped function.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        parts = [f"{key}: {_stringify(val)}" for key, val in value.items()]
        return ", ".join(part for part in parts if part and not part.endswith(": "))
    if isinstance(value, (list, tuple)):
        parts = [_stringify(item) for item in value]
        return "; ".join(part for part in parts if part)
    return str(value)


class OptimaDocumentConstructor:
    """Constructs LangChain Documents from Optima JSON files."""

    def __init__(self, representation_mode: str = "hybrid"):
        """
        Initialize document constructor.

        Args:
            representation_mode: One of "raw", "enriched", "semantic", "compiler", "hybrid"
        """
        self.representation_mode = representation_mode
        self.valid_modes = ["raw", "enriched", "semantic", "compiler", "hybrid"]
        if representation_mode not in self.valid_modes:
            raise ValueError(f"Invalid representation mode: {representation_mode}. "
                           f"Must be one of {self.valid_modes}")

    def construct_documents_from_json(self, json_path: Path,
                                    enrichment_model: Optional[str] = None) -> List[Document]:
        """
        Construct documents from a JSON file.

        Args:
            json_path: Path to JSON file
            enrichment_model: Name of enrichment model (None for raw baseline)

        Returns:
            List of LangChain Documents
        """
        logger.info(f"Loading JSON from {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)

        documents = []

        # Extract project info
        project_info = data.get("project", {})
        project_name = project_info.get("name", "unknown")
        project_root = project_info.get("root", "")
        language = project_info.get("language", "")

        # Process each file
        for file_data in data.get("files", []):
            file_path = file_data.get("path", "")
            file_name = file_data.get("name", "")
            relative_path = file_data.get("relative_path", file_path)

            # Process each function in the file
            for func_data in file_data.get("functions", []):
                doc = self._construct_function_document(
                    func_data, file_path, file_name, relative_path,
                    project_name, project_root, language, enrichment_model
                )
                if doc:
                    documents.append(doc)

        logger.info(f"Constructed {len(documents)} documents from {json_path}")
        return documents

    def _construct_function_document(self, func_data: Dict[str, Any],
                                   file_path: str, file_name: str,
                                   relative_path: str, project_name: str,
                                   project_root: str, language: str,
                                   enrichment_model: Optional[str]) -> Optional[Document]:
        """
        Construct a single document from function data.

        Args:
            func_data: Function data from JSON
            file_path: Absolute path to source file
            file_name: Name of source file
            relative_path: Relative path of source file
            project_name: Name of project
            project_root: Root directory of project
            language: Programming language
            enrichment_model: Name of enrichment model (None for raw)

        Returns:
            LangChain Document or None if construction fails
        """
        try:
            # Enrichment output stores model-generated fields under an
            # "enrichment" object; flatten them for representation building.
            nested_enrichment = func_data.get("enrichment", {})
            if isinstance(nested_enrichment, dict):
                func_data = {**func_data, **nested_enrichment}

            func_id = func_data.get("id", "")
            func_name = func_data.get("name", "")
            qualified_name = func_data.get("qualified_name", "")
            return_type = func_data.get("return_type", "")
            parameters = func_data.get("parameters", [])
            source_location = func_data.get("source_location", {})
            source_code = func_data.get("source_code", "")

            # Skip if no function ID
            if not func_id:
                return None

            # Build content based on representation mode
            content_parts = []

            # Always include basic function identification
            content_parts.append(f"Function: {func_name}")
            if qualified_name:
                content_parts.append(f"Qualified Name: {qualified_name}")
            if return_type:
                content_parts.append(f"Return Type: {return_type}")
            if parameters:
                param_strs = []
                for param in parameters:
                    param_name = param.get("name", "")
                    param_type = param.get("type", "")
                    if param_name and param_type:
                        param_strs.append(f"{param_type} {param_name}")
                    elif param_name:
                        param_strs.append(param_name)
                if param_strs:
                    content_parts.append(f"Parameters: {_stringify(param_strs)}")

            # Add source location info
            if source_location:
                file_loc = source_location.get("file", "")
                start_line = source_location.get("start_line", "")
                if file_loc:
                    content_parts.append(f"Source File: {file_loc}")
                if start_line:
                    content_parts.append(f"Start Line: {start_line}")

            # Add source code
            if source_code:
                content_parts.append(f"Source Code:\n{source_code}")

            # Add LLVM IR if available (compiler representation)
            llvm_ir = func_data.get("llvm_ir", "")
            if llvm_ir and self.representation_mode in ["compiler", "hybrid", "enriched"]:
                # Truncate LLVM IR to avoid overly long documents
                if len(llvm_ir) > 1000:
                    llvm_ir = llvm_ir[:1000] + "\n... (truncated)"
                content_parts.append(f"LLVM IR:\n{llvm_ir}")

            # Add AST info if available
            ast_json = func_data.get("ast_json", "")
            if ast_json and self.representation_mode in ["compiler", "hybrid"]:
                # Truncate AST JSON
                if len(ast_json) > 500:
                    ast_json = ast_json[:500] + "\n... (truncated)"
                content_parts.append(f"AST JSON:\n{ast_json}")

            # Add CFG if available
            cfg_dot = func_data.get("cfg_dot", "")
            if cfg_dot and self.representation_mode in ["compiler", "hybrid"]:
                if len(cfg_dot) > 500:
                    cfg_dot = cfg_dot[:500] + "\n... (truncated)"
                content_parts.append(f"CFG DOT:\n{cfg_dot}")

            # Add call graph info
            call_graph = func_data.get("call_graph", {})
            if call_graph and self.representation_mode in ["compiler", "hybrid", "enriched"]:
                calls = call_graph.get("calls", [])
                called_by = call_graph.get("called_by", [])
                if calls:
                    content_parts.append(f"Calls: {_stringify(calls)}")
                if called_by:
                    content_parts.append(f"Called By: {_stringify(called_by)}")

            # Add dependencies
            dependencies = func_data.get("dependencies", [])
            if dependencies:
                content_parts.append(f"Dependencies: {_stringify(dependencies)}")

            # Add semantic enrichment fields if available and mode allows
            if enrichment_model and self.representation_mode in ["enriched", "semantic", "hybrid"]:
                # Purpose
                purpose = func_data.get("purpose", "")
                if purpose:
                    content_parts.append(f"Purpose: {_stringify(purpose)}")

                # Behavior
                behavior = func_data.get("behavior", "")
                if behavior:
                    content_parts.append(f"Behavior: {_stringify(behavior)}")

                # Summary
                summary = func_data.get("summary", "")
                if summary:
                    content_parts.append(f"Summary: {_stringify(summary)}")

                # Inputs -- may be a flat list of strings, or (an
                # unrestricted LLM may return this) a list of structured
                # objects like [{"name": "board", "type": "Board*"}].
                inputs = func_data.get("inputs", [])
                if inputs:
                    content_parts.append(f"Inputs: {_stringify(inputs)}")

                # Outputs
                outputs = func_data.get("outputs", [])
                if outputs:
                    content_parts.append(f"Outputs: {_stringify(outputs)}")

                # Side effects
                side_effects = func_data.get("side_effects", [])
                if side_effects:
                    content_parts.append(f"Side Effects: {_stringify(side_effects)}")

                # Concepts
                concepts = func_data.get("concepts", [])
                if concepts:
                    content_parts.append(f"Concepts: {_stringify(concepts)}")

                # Keywords
                keywords = func_data.get("keywords", [])
                if keywords:
                    content_parts.append(f"Keywords: {_stringify(keywords)}")

                # Algorithm
                algorithm = func_data.get("algorithm", "")
                if algorithm:
                    content_parts.append(f"Algorithm: {_stringify(algorithm)}")

                # Complexity -- normally {"time": ..., "space": ...}, but
                # rendered safely even if the model returned something else.
                complexity = func_data.get("complexity", {})
                if complexity:
                    if isinstance(complexity, dict):
                        time_complexity = complexity.get("time", "")
                        space_complexity = complexity.get("space", "")
                        if time_complexity:
                            content_parts.append(f"Time Complexity: {_stringify(time_complexity)}")
                        if space_complexity:
                            content_parts.append(f"Space Complexity: {_stringify(space_complexity)}")
                    else:
                        content_parts.append(f"Complexity: {_stringify(complexity)}")

            # For semantic-only mode, focus primarily on semantic content
            if self.representation_mode == "semantic":
                semantic_parts = []
                # Keep only semantic-enriched content for semantic-only mode
                purpose = func_data.get("purpose", "")
                if purpose:
                    semantic_parts.append(f"Purpose: {_stringify(purpose)}")

                behavior = func_data.get("behavior", "")
                if behavior:
                    semantic_parts.append(f"Behavior: {_stringify(behavior)}")

                summary = func_data.get("summary", "")
                if summary:
                    semantic_parts.append(f"Summary: {_stringify(summary)}")

                concepts = func_data.get("concepts", [])
                if concepts:
                    semantic_parts.append(f"Concepts: {_stringify(concepts)}")

                keywords = func_data.get("keywords", [])
                if keywords:
                    semantic_parts.append(f"Keywords: {_stringify(keywords)}")

                algorithm = func_data.get("algorithm", "")
                if algorithm:
                    semantic_parts.append(f"Algorithm: {_stringify(algorithm)}")

                complexity = func_data.get("complexity", {})
                if complexity:
                    if isinstance(complexity, dict):
                        time_complexity = complexity.get("time", "")
                        space_complexity = complexity.get("space", "")
                        if time_complexity:
                            semantic_parts.append(f"Time Complexity: {_stringify(time_complexity)}")
                        if space_complexity:
                            semantic_parts.append(f"Space Complexity: {_stringify(space_complexity)}")
                    else:
                        semantic_parts.append(f"Complexity: {_stringify(complexity)}")

                # If we have semantic content, use it; otherwise fall back to basic info
                if semantic_parts:
                    content_parts = semantic_parts
                # If no semantic content, keep minimal identification
                elif not any([purpose, behavior, summary, concepts, keywords, algorithm]):
                    content_parts = [f"Function: {func_name}", f"Qualified Name: {qualified_name}"]

            # Join all content parts
            page_content = "\n\n".join(content_parts)

            # Prepare metadata
            metadata = {
                "function_id": func_id,
                "function_name": func_name,
                "qualified_name": qualified_name,
                "return_type": return_type,
                "file_path": file_path,
                "file_name": file_name,
                "relative_path": relative_path,
                "project_name": project_name,
                "project_root": project_root,
                "language": language,
                "source_location": source_location,
                "corpus_type": "enriched" if enrichment_model else "raw",
                "enrichment_model": enrichment_model if enrichment_model else "none",
                "representation_mode": self.representation_mode
            }

            # Add complexity and category info if available
            metadata["complexity_category"] = self._determine_complexity_category(func_data)
            metadata["function_type"] = self._determine_function_type(func_data)
            metadata["graph_complexity"] = self._determine_graph_complexity(func_data)
            _dependencies = func_data.get("dependencies")
            metadata["dependency_count"] = len(_dependencies) if isinstance(_dependencies, (list, dict, str)) else 0

            # Create document
            document = Document(
                page_content=page_content,
                metadata=metadata
            )

            return document

        except Exception as e:
            logger.error(f"Error constructing document for function {func_data.get('id', 'unknown')}: {e}")
            # Never drop a base function from the corpus over an
            # unexpected enrichment-field shape: fall back to a minimal,
            # base-only document (identity + source) rather than returning
            # None, so "one document per base function" holds even when
            # something about this function's data could not be rendered
            # as intended. The error is recorded in metadata, not hidden.
            fallback_id = func_data.get("id", "")
            if not fallback_id:
                return None
            fallback_content = "\n\n".join(part for part in [
                f"Function: {func_data.get('name', '')}",
                f"Qualified Name: {func_data.get('qualified_name', '')}",
                f"Source Code:\n{func_data.get('source_code')}" if func_data.get("source_code") else "",
            ] if part)
            return Document(
                page_content=fallback_content or f"Function: {func_data.get('name', '')}",
                metadata={
                    "function_id": fallback_id,
                    "function_name": func_data.get("name", ""),
                    "qualified_name": func_data.get("qualified_name", ""),
                    "file_path": file_path,
                    "file_name": file_name,
                    "relative_path": relative_path,
                    "project_name": project_name,
                    "project_root": project_root,
                    "language": language,
                    "corpus_type": "enriched" if enrichment_model else "raw",
                    "enrichment_model": enrichment_model if enrichment_model else "none",
                    "representation_mode": self.representation_mode,
                    "document_construction_error": str(e),
                },
            )

    def _determine_complexity_category(self, func_data: Dict[str, Any]) -> str:
        """Determine complexity category based on available metrics."""
        # Try to get from existing complexity metrics
        complexity = func_data.get("complexity", {})
        if complexity:
            # Could analyze time/space complexity strings
            # For now, use a simple heuristic based on parameter count and dependencies
            pass

        # Heuristic based on available data
        param_count = len(func_data.get("parameters", []))
        dependency_count = len(func_data.get("dependencies", []))
        source_code = func_data.get("source_code", "")
        lines = len(source_code.split('\n')) if source_code else 0

        # Simple scoring
        score = param_count + (dependency_count * 2) + (lines // 10)

        if score <= 5:
            return "Simple"
        elif score <= 15:
            return "Moderate"
        else:
            return "Complex"

    def _determine_function_type(self, func_data: Dict[str, Any]) -> str:
        """Determine function type from qualified name or other indicators."""
        qualified_name = func_data.get("qualified_name", "")
        func_name = func_data.get("name", "")
        return_type = func_data.get("return_type", "")

        # Check for common patterns
        if "::" in qualified_name:
            if "Test" in qualified_name or "test" in func_name.lower():
                return "Test"
            elif "Constructor" in qualified_name or func_name == qualified_name.split("::")[-1]:
                # Simple heuristic: if function name matches class name, likely constructor
                class_name = qualified_name.split("::")[-2] if len(qualified_name.split("::")) >= 2 else ""
                if class_name and func_name == class_name:
                    return "Constructor"
            elif "Destructor" in qualified_name or "~" in func_name:
                return "Destructor"
            elif "Method" in qualified_name or "." in qualified_name:
                return "Method"
            else:
                return "Function"
        else:
            # Global function or simple case
            if "test" in func_name.lower():
                return "Test"
            else:
                return "Function"

    def _determine_graph_complexity(self, func_data: Dict[str, Any]) -> str:
        """Determine graph complexity based on call graph."""
        call_graph = func_data.get("call_graph", {})
        calls = len(call_graph.get("calls", []))
        called_by = len(call_graph.get("called_by", []))
        total_connections = calls + called_by

        if total_connections == 0:
            return "Leaf"
        elif total_connections <= 3:
            return "Low"
        elif total_connections <= 7:
            return "Medium"
        else:
            return "High"


def discover_json_files(output_dir: Path) -> Dict[str, Path]:
    """
    Discover all JSON files in the output directory.

    Args:
        output_dir: Path to output directory

    Returns:
        Dictionary mapping model names to file paths, with 'base' for base.json
    """
    json_files = {}

    # Find base.json
    base_path = output_dir / "base.json"
    if base_path.exists():
        json_files["base"] = base_path

    # Find enhanced JSON files
    for json_file in output_dir.glob("enhanced_*.json"):
        # Extract model name from filename
        # Format: enhanced_<model-name>.json
        stem = json_file.stem  # removes .json extension
        if stem.startswith("enhanced_"):
            model_name = stem[9:]  # remove 'enhanced_' prefix
            json_files[model_name] = json_file

    logger = logging.getLogger(__name__)
    logger.info(f"Discovered {len(json_files)} JSON files: {list(json_files.keys())}")
    return json_files


if __name__ == "__main__":
    # Test the document constructor
    logging.basicConfig(level=logging.INFO)

    output_dir = Path("/home/igorr/Projects/op-python/optima/output")
    json_files = discover_json_files(output_dir)

    print(f"Found JSON files: {list(json_files.keys())}")

    # Test with base.json
    if "base" in json_files:
        constructor = OptimaDocumentConstructor(representation_mode="hybrid")
        docs = constructor.construct_documents_from_json(json_files["base"], enrichment_model=None)
        print(f"Constructed {len(docs)} documents from base.json")
        if docs:
            print(f"Sample document metadata: {docs[0].metadata}")
            print(f"Sample document content preview: {docs[0].page_content[:200]}...")