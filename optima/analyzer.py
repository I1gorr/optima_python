"""
Optima analyzer: extracts compiler-derived information from C/C++ projects.
"""
import ctypes.util
import glob
import json
import os
import shutil
import subprocess
import tempfile
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import clang.cindex
from clang.cindex import CursorKind, TypeKind, StorageClass


def safe_slug(name: str) -> str:
    """Turn a project/directory name into a stable, filesystem- and
    corpus-safe identifier (used as the test-suite/dataset name so base.json
    outputs for different test suites never collide). Matches the slug
    convention already used for model/embedding identifiers elsewhere in
    Optima (``optima.rag.embedding_simple._slug``).
    """
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(name).strip()).strip("-").lower()
    return slug or "project"


def _find_libclang() -> Optional[Path]:
    """Locate the native libclang shared library without assuming a version.

    Debian/Ubuntu packages install libclang under a version-specific path
    (e.g. /usr/lib/llvm-14/lib/libclang.so.1) rather than the unversioned
    /usr/lib/libclang.so the 'clang' PyPI bindings look for by default, and the
    exact filename changes with every LLVM release. Pinning one literal
    filename (e.g. libclang.so.22.1.8) breaks the moment the installed LLVM
    version differs from whatever was hardcoded, so every candidate here is
    discovered dynamically, never hardcoded to a specific version.
    """
    configured = os.environ.get("LIBCLANG_PATH")
    if configured:
        configured_path = Path(configured)
        direct = configured_path / "libclang.so" if configured_path.is_dir() else configured_path
        if direct.exists():
            return direct

    # Some 'clang'/'libclang' PyPI packages bundle their own precompiled native
    # library right next to the Python bindings (clang/native/libclang.so).
    # When present this is the best candidate: it is guaranteed to match the
    # exact Python package version actually imported, avoiding any Python
    # binding / native library version mismatch entirely.
    bundled = Path(clang.cindex.__file__).resolve().parent / "native" / "libclang.so"
    if bundled.exists():
        return bundled

    found = ctypes.util.find_library("clang")
    if found:
        return Path(found)

    patterns = [
        "/usr/lib/llvm-*/lib/libclang.so*",
        "/usr/lib/*/libclang-*.so*",  # e.g. /usr/lib/x86_64-linux-gnu/libclang-14.so.1
        "/usr/lib/*/libclang.so*",
        "/usr/lib/libclang*.so*",
        "/usr/local/lib/libclang*.so*",
    ]
    candidates = [Path(p) for pattern in patterns for p in glob.glob(pattern) if Path(p).is_file()]
    if not candidates:
        return None

    def _version_key(path: Path) -> int:
        match = re.search(r"llvm-(\d+)", str(path)) or re.search(r"(\d+)", path.name)
        return int(match.group(1)) if match else -1

    # Prefer the newest LLVM version when several are installed side by side.
    candidates.sort(key=_version_key)
    return candidates[-1]


def _configure_clang() -> None:
    """Load libclang on local, Colab, and Kaggle installations."""
    if clang.cindex.Config.loaded:
        return
    library = _find_libclang()
    if library:
        clang.cindex.Config.set_library_file(str(library))
    # If nothing was found, fall through and let clang's own discovery produce
    # its actionable error rather than silently failing here.


_configure_clang()

# Free functions, constructors, and destructors are the historically-handled
# kinds. Regular/static/virtual class methods (CXX_METHOD), function templates,
# and user-defined conversion operators are function-like too and must be
# extracted and resolved as call targets the same way, or Optima silently
# drops most of the API surface of any object-oriented C++ codebase.
FUNCTION_LIKE_KINDS = (
    CursorKind.FUNCTION_DECL, CursorKind.CONSTRUCTOR, CursorKind.DESTRUCTOR,
    CursorKind.CXX_METHOD, CursorKind.FUNCTION_TEMPLATE, CursorKind.CONVERSION_FUNCTION,
)
CLASS_LIKE_KINDS = (
    CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL, CursorKind.CLASS_TEMPLATE,
)
NAME_CONTRIBUTING_KINDS = FUNCTION_LIKE_KINDS + CLASS_LIKE_KINDS + (
    CursorKind.UNION_DECL, CursorKind.NAMESPACE, CursorKind.TYPEDEF_DECL,
    CursorKind.ENUM_DECL,
)


def _enclosing_class_cursor(cursor: clang.cindex.Cursor) -> Optional[clang.cindex.Cursor]:
    """Walk up to the nearest enclosing class/struct/class-template definition."""
    parent = cursor.semantic_parent
    while parent:
        if parent.kind in CLASS_LIKE_KINDS:
            return parent
        parent = parent.semantic_parent
    return None


def _class_id_for_cursor(cursor: clang.cindex.Cursor, project_root: str) -> Optional[str]:
    """Build a class id from a class cursor. Derived purely from the cursor's own
    identity so a method computes the same id as the class's own extraction pass."""
    if cursor.location.file is None:
        return None
    relative_path = os.path.relpath(
        os.path.abspath(os.path.realpath(cursor.location.file.name)),
        os.path.abspath(os.path.realpath(project_root)),
    )
    qualified_name = _qualified_cursor_name(cursor)
    return f"class::{relative_path}::{qualified_name}::{cursor.location.line}"


class FunctionInfo:
    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        file_path: str,
        project_root: str,
        include_dirs: Optional[List[str]] = None,
        compile_cache: Optional[Dict[str, Dict[str, str]]] = None
    ):
        self.cursor = cursor
        cursor_file = cursor.location.file.name if cursor.location.file else file_path
        self.file_path = str(Path(cursor_file).resolve())
        self.project_root = str(Path(project_root).resolve())
        self.include_dirs = include_dirs or _discover_include_dirs(Path(self.project_root))
        self.compile_cache = compile_cache if compile_cache is not None else {}
        self.relative_path = os.path.relpath(self.file_path, self.project_root)
        self.qualified_name = self._get_qualified_name()
        self.name = cursor.spelling
        self.mangled_name = getattr(cursor, "mangled_name", "") or ""
        self.id = self._generate_id()
        self.return_type = self._get_return_type()
        self.parameters = self._get_parameters()
        self.source_location = self._get_source_location()
        self.source_code = self._get_source_code()
        enclosing_class = _enclosing_class_cursor(cursor)
        self.class_id = (
            _class_id_for_cursor(enclosing_class, self.project_root)
            if enclosing_class is not None else None
        )
        self.class_info = {
            "class_id": self.class_id,
            "is_method": enclosing_class is not None,
            "is_constructor": cursor.kind == CursorKind.CONSTRUCTOR,
            "is_destructor": cursor.kind == CursorKind.DESTRUCTOR,
            "is_static": cursor.is_static_method(),
            "is_virtual": cursor.is_virtual_method(),
            "is_pure_virtual": cursor.is_pure_virtual_method(),
            "is_const": cursor.is_const_method(),
            "is_template": cursor.kind == CursorKind.FUNCTION_TEMPLATE,
            "is_conversion_operator": cursor.kind == CursorKind.CONVERSION_FUNCTION,
            "is_operator_overload": self.name.startswith("operator") and self.name != "operator",
            "access_specifier": (
                cursor.access_specifier.name
                if enclosing_class is not None else None
            ),
        }
        self.analysis_status = "pending"
        self.compiler_error = ""
        self.llvm_ir = ""
        self.llvm_function_name = ""
        self.ast = self._get_ast()
        self.basic_blocks = []
        self.cfg = {"entry_node": "", "exit_node": "", "nodes": [], "edges": []}
        self.llvm_status = "not_attempted"
        self.calls = []  # List of function IDs called
        self.called_by = []  # To be filled later
        self.dependencies = []  # List of dependencies (function, class, global, external)

    def _generate_id(self) -> str:
        # <relative_path>::<qualified_function_name>::<start_line>
        start_line = self.cursor.location.line
        return f"{self.relative_path}::{self.qualified_name}::{start_line}"

    def _get_qualified_name(self) -> str:
        # Get the qualified name by traversing the semantic parent
        name_parts = []
        cursor = self.cursor
        while cursor:
            if cursor.kind in NAME_CONTRIBUTING_KINDS:
                if cursor.spelling:
                    name_parts.insert(0, cursor.spelling)
            cursor = cursor.semantic_parent
        return "::".join(name_parts)

    def _get_return_type(self) -> str:
        return self.cursor.result_type.spelling

    def _get_parameters(self) -> List[Dict[str, str]]:
        params = []
        for arg in self.cursor.get_arguments():
            params.append({
                "name": arg.spelling,
                "type": arg.type.spelling
            })
        return params

    def _get_source_location(self) -> Dict[str, Any]:
        start = self.cursor.location
        end = self.cursor.extent.end
        return {
            "file": self.cursor.location.file.name if self.cursor.location.file else "",
            "start_line": start.line,
            "start_column": start.column,
            "end_line": end.line,
            "end_column": end.column
        }

    def _get_source_code(self) -> str:
        if not self.cursor.location.file:
            return ""
        with open(self.cursor.location.file.name, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        start_line = self.cursor.location.line - 1  # 0-indexed
        end_line = self.cursor.extent.end.line - 1
        return ''.join(lines[start_line:end_line+1])

    def enrich_with_llvm(self, llvm_ir: str, status: str = "compiled", error: str = ""):
        self.llvm_ir = llvm_ir
        self.llvm_status = status
        self.compiler_error = error
        if status != "compiled":
            self.analysis_status = "source_only"
            return
        self._build_cfg()

    def _get_ast(self) -> Dict[str, Any]:
        # Return a simplified AST representation for the function
        return self._cursor_to_dict(self.cursor)

    def _cursor_to_dict(self, cursor: clang.cindex.Cursor) -> Dict[str, Any]:
        """Convert a cursor and its children to a simple dict representation."""
        node = {
            "kind": cursor.kind.name,
            "spelling": cursor.spelling,
            "displayname": cursor.displayname,
            "location": {
                "file": cursor.location.file.name if cursor.location.file else None,
                "line": cursor.location.line,
                "column": cursor.location.column
            } if cursor.location.file else None,
            "children": []
        }
        for child in cursor.get_children():
            node["children"].append(self._cursor_to_dict(child))
        return node

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "mangled_name": self.mangled_name,
            "class_info": self.class_info,
            "return_type": self.return_type,
            "parameters": self.parameters,
            "source_location": self.source_location,
            "source_code": self.source_code,
            "llvm_ir": self.llvm_ir,
            "analysis_status": self.analysis_status,
            "compiler_error": self.compiler_error,
            "source": {
                "file": self.relative_path,
                "start_line": self.source_location["start_line"],
                "end_line": self.source_location["end_line"],
                "start_column": self.source_location["start_column"],
                "end_column": self.source_location["end_column"],
            },
            "llvm": {
                "matched": self.analysis_status == "success",
                "status": self.llvm_status,
                **({"function_name": self.llvm_function_name} if self.llvm_function_name else {}),
                **({"mangled_name": self.mangled_name} if self.mangled_name else {}),
                **({"basic_blocks": self.basic_blocks, "cfg": self.cfg}
                   if self.analysis_status == "success" else {}),
                **({"reason": "not_emitted_or_not_found"}
                   if self.llvm_status == "compiled" and self.analysis_status != "success"
                   else {}),
                **({"error": self.compiler_error} if self.llvm_status == "compilation_failed" else {}),
            },
            "ast": self.ast,
            "basic_blocks": self.basic_blocks,
            "cfg": self.cfg,
            "calls": self.calls,
            "called_by": self.called_by,
            "dependencies": self.dependencies
        }


    def _build_cfg(self):
        """Build control-flow graph from LLVM IR.

        self.llvm_ir starts out holding the *entire translation unit's* IR
        (the same string is handed to every function extracted from that
        file). It is narrowed to just this function's own block below, on
        every exit path: left as the whole file it would be duplicated once
        per function in the final base.json (a file with 30 functions would
        write ~30 copies of its own IR), which is exactly the kind of
        multi-GB-output/terminal-flooding blowup a project the size of
        OpenSSL hits immediately.
        """
        llvm_ir = self.llvm_ir
        if not llvm_ir:
            return

        function_match = self._find_llvm_function(llvm_ir)
        if function_match is None:
            self.llvm_ir = ""
            self.basic_blocks = []
            self.cfg = {"entry_node": "", "exit_node": "", "nodes": [], "edges": []}
            self.analysis_status = "llvm_function_not_found"
            self.compiler_error = (
                f"Function '{self.mangled_name or self.qualified_name}' "
                "was not found in generated LLVM IR."
            )
            return

        match, llvm_name = function_match
        self.llvm_function_name = llvm_name
        func_body = self._extract_llvm_function_body(llvm_ir, match.end())
        if func_body is None:
            self.llvm_ir = ""
            self.basic_blocks = []
            self.cfg = {"entry_node": "", "exit_node": "", "nodes": [], "edges": []}
            self.analysis_status = "llvm_function_not_found"
            self.compiler_error = "Generated LLVM function body is malformed."
            return

        # Narrow self.llvm_ir to just this function (signature + body) now
        # that it has been located, instead of leaving the whole file on it.
        self.llvm_ir = llvm_ir[match.start():match.end()] + func_body + "\n}"

        blocks = self._parse_llvm_blocks(func_body)
        if not blocks:
            self.basic_blocks = []
            self.cfg = {"entry_node": "", "exit_node": "", "nodes": [], "edges": []}
            self.analysis_status = "llvm_function_not_found"
            self.compiler_error = "Generated LLVM function contains no basic blocks."
            return

        debug_lines = self._extract_debug_lines(llvm_ir)
        for block in blocks:
            lines = [
                debug_lines[metadata_id]
                for instruction in block["instructions"]
                for metadata_id in re.findall(r"!dbg\s+!(\d+)", instruction)
                if metadata_id in debug_lines
            ]
            if lines:
                block["source_location_available"] = True
                block["start_line"] = min(lines)
                block["end_line"] = max(lines)
            else:
                block["source_location_available"] = False

        edges = []
        for index, block in enumerate(blocks):
            terminator = self._get_terminator(block["instructions"])
            if terminator is None:
                targets = [blocks[index + 1]["id"]] if index + 1 < len(blocks) else []
            else:
                targets = self._get_terminator_targets(terminator)
            edges.extend({"from": block["id"], "to": target} for target in targets)

        entry_node = blocks[0]["id"]
        exit_nodes = []
        for block in blocks:
            terminator = self._get_terminator(block["instructions"])
            if terminator and terminator.startswith("ret "):
                exit_nodes.append(block["id"])
        exit_node = exit_nodes[-1] if exit_nodes else blocks[-1]["id"]
        self.basic_blocks = blocks
        self.cfg = {
            "entry_node": entry_node,
            "exit_node": exit_node,
            "nodes": [block["id"] for block in blocks],
            "edges": edges
        }
        self.analysis_status = "success"

    def _find_llvm_function(self, llvm_ir: str):
        """Find a definition by its exact LLVM symbol name."""
        candidates = [self.mangled_name, self.qualified_name, self.name]
        candidates = [candidate for candidate in candidates if candidate]
        definition_pattern = re.compile(
            r"^\s*define\b[^{\n]*@"
            r"(?:(?P<quoted>\"[^\"]+\")|(?P<plain>[^\s(]+))"
            r"[^{\n]*\{",
            re.MULTILINE
        )
        for match in definition_pattern.finditer(llvm_ir):
            llvm_name = match.group("quoted") or match.group("plain")
            llvm_name = llvm_name.strip('"')
            if llvm_name in candidates:
                return match, llvm_name
        return None

    @staticmethod
    def _extract_llvm_function_body(llvm_ir: str, body_start: int) -> Optional[str]:
        """Extract a function body using LLVM's closing-brace line."""
        body_end = re.search(r"(?m)^\s*}\s*$", llvm_ir[body_start:])
        if body_end is None:
            return None
        return llvm_ir[body_start:body_start + body_end.start()]

    @staticmethod
    def _parse_llvm_blocks(func_body: str) -> List[Dict[str, Any]]:
        blocks = []
        current = {"id": "entry", "name": "entry", "instructions": []}
        for line in func_body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                continue
            label_match = re.fullmatch(
                r"(?P<label>[-A-Za-z0-9$._]+):(?:\s*;.*)?", stripped
            )
            if label_match:
                if not blocks and current["id"] == "entry" and not current["instructions"]:
                    label = label_match.group("label")
                    current = {"id": label, "name": label, "instructions": []}
                    continue
                if current["instructions"]:
                    blocks.append(current)
                label = label_match.group("label")
                current = {"id": label, "name": label, "instructions": []}
            else:
                current["instructions"].append(stripped)
        if current["instructions"] or not blocks:
            blocks.append(current)
        return blocks

    @staticmethod
    def _get_terminator(instructions: List[str]) -> Optional[str]:
        for instruction in reversed(instructions):
            if re.match(r"^(br|switch|ret|invoke|unreachable)\b", instruction):
                return instruction
        return None

    @staticmethod
    def _get_terminator_targets(terminator: str) -> List[str]:
        if terminator.startswith("ret ") or terminator.startswith("unreachable"):
            return []
        return re.findall(r"label\s+%?([^,\s\]]+)", terminator)

    @staticmethod
    def _extract_debug_lines(llvm_ir: str) -> Dict[str, int]:
        return {
            metadata_id: int(line)
            for metadata_id, line in re.findall(
                r"!(\d+)\s*=\s*!DILocation\(line:\s*(\d+)", llvm_ir
            )
        }


def _discover_include_dirs(project_path: Path, exclude_dirs: Optional[Set[str]] = None) -> List[str]:
    """Discover absolute project directories that may contain included headers.

    Only the project root and directories literally named 'include'/'src' are
    added as global -I search roots. A same-directory quoted #include already
    resolves without any -I entry (the preprocessor always searches the
    including file's own directory first), so granting every header-bearing
    subdirectory its own global -I root is unnecessary and actively harmful:
    on a large project it lets a deeply-nested internal header shadow a
    same-named standard-library header for unrelated files elsewhere in the
    tree (e.g. OpenSSL's include/internal/time.h — meant to be reached only
    as "internal/time.h" relative to the include/ root — was matching bare
    #include <time.h> in other files once include/internal itself was added
    as a top-level -I root, silently corrupting struct timeval).
    """
    project_path = project_path.resolve()
    include_dirs = {str(project_path)}
    exclude_dirs = exclude_dirs or set()

    for root, dirs, files in os.walk(project_path):
        dirs[:] = [d for d in dirs if d.lower() not in exclude_dirs]
        root_path = Path(root).resolve()
        if root_path.name.lower() in {'include', 'src'}:
            include_dirs.add(str(root_path))

    return sorted(include_dirs)


def _is_project_owned(cursor: clang.cindex.Cursor, project_root: str) -> bool:
    """Return whether a cursor's source file is inside the analyzed project."""
    location_file = cursor.location.file
    if location_file is None:
        return False
    source_path = os.path.abspath(os.path.realpath(location_file.name))
    project_path = os.path.abspath(os.path.realpath(project_root))
    try:
        return os.path.commonpath([source_path, project_path]) == project_path
    except ValueError:
        return False


_COMPILER_PATH_CACHE: Dict[str, Optional[str]] = {}
_RESOURCE_DIR_CACHE: Dict[str, Optional[str]] = {}


def _find_clang_compiler(name: str) -> Optional[str]:
    """Locate a clang/clang++ executable without assuming an exact version.

    Prefers an unversioned entry on PATH (e.g. /usr/bin/clang++), and falls
    back to the versioned binaries Debian/Ubuntu clang packages install
    (clang++-14, clang++-18, ...) when no unversioned one is on PATH, picking
    the newest version found. Never hardcodes a specific LLVM release.
    """
    if name in _COMPILER_PATH_CACHE:
        return _COMPILER_PATH_CACHE[name]
    resolved = shutil.which(name)
    if resolved is None:
        def _version_key(p: str) -> int:
            match = re.search(r"llvm-(\d+)", p) or re.search(r"-(\d+)$", p)
            return int(match.group(1)) if match else -1

        versioned = sorted(
            glob.glob(f"/usr/bin/{name}-*") + glob.glob(f"/usr/lib/llvm-*/bin/{name}"),
            key=_version_key,
        )
        resolved = versioned[-1] if versioned else None
    _COMPILER_PATH_CACHE[name] = resolved
    return resolved


def _clang_resource_dir(compiler: str) -> Optional[str]:
    """Ask the compiler itself for its resource directory, which is where its
    builtin headers (stddef.h, stdarg.h, stdbool.h, ...) live.

    When more than one clang/LLVM install is present on a machine, the
    'clang++' actually invoked can end up disagreeing with whichever
    resource directory the system's default header search picks up,
    producing errors like "fatal error: 'stddef.h' file not found" even
    though a perfectly good stddef.h exists elsewhere on disk. Asking this
    exact compiler binary via `-print-resource-dir` and pinning it with
    -resource-dir keeps header resolution tied to the compiler actually
    running, instead of relying on ambient auto-detection.
    """
    if compiler in _RESOURCE_DIR_CACHE:
        return _RESOURCE_DIR_CACHE[compiler]
    resource_dir = None
    try:
        result = subprocess.run(
            [compiler, "-print-resource-dir"], capture_output=True, text=True, timeout=10
        )
        candidate = result.stdout.strip()
        if result.returncode == 0 and candidate:
            candidate_path = Path(candidate)
            # Most LLVM packaging puts builtin headers under <resource-dir>/include/,
            # but some distributions (observed on Kaggle's LLVM install) put them
            # directly under <resource-dir>/. Accept either layout rather than
            # assuming one.
            if (candidate_path / "include" / "stddef.h").exists() or (candidate_path / "stddef.h").exists():
                resource_dir = candidate
    except (OSError, subprocess.TimeoutExpired):
        pass
    _RESOURCE_DIR_CACHE[compiler] = resource_dir
    return resource_dir


def _compile_translation_unit(
    file_path: Path,
    include_dirs: List[str],
    compile_cache: Dict[str, Dict[str, str]]
) -> Dict[str, str]:
    path = str(file_path.resolve())
    if path in compile_cache:
        return compile_cache[path]
    is_cxx = file_path.suffix.lower() in {'.cpp', '.cc', '.cxx', '.hpp', '.hh', '.hxx'}
    compiler_name = 'clang++' if is_cxx else 'clang'
    compiler = _find_clang_compiler(compiler_name)
    if compiler is None:
        compiled = {
            "ir": "", "status": "compilation_failed",
            "error": f"No '{compiler_name}' executable was found (checked PATH and "
                     f"/usr/bin/{compiler_name}-*).",
        }
        compile_cache[path] = compiled
        return compiled
    standard = '-std=c++17' if is_cxx else '-std=c11'
    cmd = [compiler, '-S', '-emit-llvm', '-O0', '-g', standard]
    resource_dir = _clang_resource_dir(compiler)
    if resource_dir:
        cmd.extend(['-resource-dir', resource_dir])
    for include_dir in include_dirs:
        cmd.extend(['-I', include_dir])
    cmd.extend([path, '-o', '-'])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        result = None
    if result is None:
        compiled = {"ir": "", "status": "compilation_failed", "error": "Compiler timed out after 30 seconds."}
    elif result.returncode != 0:
        compiled = {"ir": "", "status": "compilation_failed", "error": result.stderr.strip()}
    else:
        compiled = {"ir": result.stdout, "status": "compiled", "error": ""}
    compile_cache[path] = compiled
    return compiled


def _finalize_and_write(
    output_file: Path,
    project_path: Path,
    processed_files: List[Path],
    all_functions: List["FunctionInfo"],
    function_map: Dict[str, "FunctionInfo"],
    functions_by_file: Dict[str, List["FunctionInfo"]],
    class_cursor_map: Dict[str, clang.cindex.Cursor],
) -> None:
    """Resolve calls/called_by/classes from whatever has been parsed so far
    and atomically (re)write base.json.

    Safe to call repeatedly as a checkpoint during a long run, not just once
    at the end: a call to a function in a file not yet parsed simply stays
    unresolved until a later checkpoint sees it. Writing base.json only once,
    after every file in a multi-thousand-file project has already been
    parsed and compiled, means a crash on the very last file (an OOM kill,
    a terminal dying, anything that isn't a clean Python exception) loses
    every function extracted up to that point. Calling this after each file
    instead means only the file being parsed when a crash happens is lost.

    The write itself is to a temp file followed by os.replace so a crash
    mid-write never leaves a truncated, unparseable base.json on disk.
    """
    for func in all_functions:
        func.calls = _extract_calls(func.cursor, function_map, str(project_path))
        func.dependencies = _extract_dependencies(func.cursor, project_path)

    called_by_map: Dict[str, List[Dict[str, Any]]] = {}
    for func in all_functions:
        for call in func.calls:
            if call.get("resolved") and call.get("id"):
                called_by_map.setdefault(call["id"], []).append({
                    "id": func.id, "name": func.name, "qualified_name": func.qualified_name,
                })
    for func in all_functions:
        func.called_by = called_by_map.get(func.id, [])

    method_ids_by_class: Dict[str, List[str]] = {}
    for func in all_functions:
        if func.class_id:
            method_ids_by_class.setdefault(func.class_id, []).append(func.id)

    classes_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for class_id, class_cursor in class_cursor_map.items():
        class_dict = _class_cursor_to_dict(
            class_cursor, class_id, str(project_path), method_ids_by_class.get(class_id, [])
        )
        classes_by_file.setdefault(class_dict["source_location"]["file"], []).append(class_dict)

    all_files = []
    for file_path in processed_files:
        relative_path = os.path.relpath(file_path, project_path)
        file_info = {
            "id": f"file::{relative_path}",
            "path": relative_path,
            "name": file_path.name,
            "relative_path": relative_path,
            "language": "cpp" if file_path.suffix.lower() in {'.cpp', '.cc', '.cxx', '.hpp', '.hh', '.hxx'} else "c",
            "functions": [f.to_dict() for f in functions_by_file.get(relative_path, [])],
            "classes": classes_by_file.get(relative_path, []),
        }
        all_files.append(file_info)

    call_graph_nodes = []
    call_graph_edges = []
    for func in all_functions:
        call_graph_nodes.append({"id": func.id, "name": func.name, "qualified_name": func.qualified_name})
        for call in func.calls:
            if call.get("resolved"):
                call_graph_edges.append({"from": func.id, "to": call["id"]})

    call_graph = {"nodes": call_graph_nodes, "edges": call_graph_edges}

    base_json = {
        "project": {
            "name": project_path.name,
            # Stable identifier for this test-suite/project directory, used
            # downstream (optima_kaggle.snapshot) to keep each test suite's
            # base.json and experiment outputs in their own directory rather
            # than an absolute path, which is not a safe/portable identifier.
            "test_suite": safe_slug(project_path.name),
            "root": str(project_path),
            "language": "cpp"
        },
        "files": all_files,
        "graph": {
            "function_nodes": [
                {
                    "id": func.id,
                    "type": "function",
                    "name": func.name,
                    "qualified_name": func.qualified_name,
                    "file": func.relative_path,
                    "start_line": func.source_location["start_line"],
                    "end_line": func.source_location["end_line"],
                }
                for func in all_functions
            ],
            "call_edges": call_graph_edges,
        },
        "call_graph": call_graph
    }

    tmp_path = output_file.with_suffix(output_file.suffix + ".tmp")
    with open(tmp_path, 'w') as f:
        json.dump(base_json, f, indent=2)
    os.replace(tmp_path, output_file)


def analyze_project(
    project_path: Path,
    output_dir: Path,
    verbose: bool = False,
    exclude_dirs: Optional[List[str]] = None,
    checkpoint_every: int = 20,
) -> Path:
    """Analyze a C/C++ project and generate base.json.

    By default, progress is a single updating line and per-file parse
    diagnostics are collapsed to an error count — on a project with
    thousands of files (e.g. OpenSSL), printing a full line per file plus
    every clang diagnostic produces megabytes of terminal output, which is
    enough to make some terminal emulators (VS Code/VSCodium's integrated
    terminal included) hang or crash well before the run finishes. Pass
    verbose=True to restore the full per-file/per-diagnostic output.

    exclude_dirs prunes any directory whose bare name matches (case
    insensitive) at any depth under project_path -- e.g. ["test", "fuzz",
    "demos"] to skip OpenSSL's test suite, fuzzers, and demo programs and
    analyze only the library itself.
    """
    project_path = project_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    exclude_names = {d.lower() for d in (exclude_dirs or [])}

    index = clang.cindex.Index.create()
    include_dirs = _discover_include_dirs(project_path, exclude_names)
    compile_cache: Dict[str, Dict[str, str]] = {}
    clang_args = ['-std=c++17']
    # libclang's own header search can disagree with the system's default one
    # (the same "fatal error: 'stddef.h' file not found" problem as the LLVM
    # compile step below, but here it aborts index.parse() entirely and loses
    # every function in the file rather than just the LLVM mapping for one —
    # so pin it to a real resource directory whenever one can be detected.
    _parse_resource_dir = _clang_resource_dir(_find_clang_compiler('clang++') or 'clang++')
    if _parse_resource_dir:
        clang_args.extend(['-resource-dir', _parse_resource_dir])
    for include_dir in include_dirs:
        clang_args.extend(['-I', include_dir])

    # Find all relevant files
    extensions = {'.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hh', '.hxx'}
    source_files = []
    for root, dirs, files in os.walk(project_path):
        dirs[:] = [d for d in dirs if d.lower() not in exclude_names]
        for file in files:
            if Path(file).suffix in extensions:
                source_files.append(Path(root) / file)

    # We'll store all functions and files
    all_functions = []  # Flat list of all functions for call graph
    function_map = {}  # Map from function ID to FunctionInfo
    functions_by_file: Dict[str, List[FunctionInfo]] = {}
    class_cursor_map: Dict[str, clang.cindex.Cursor] = {}  # class id -> defining cursor
    excluded_system_functions = set()
    processed_files: List[Path] = []

    output_file = output_dir / "base.json"
    total_files = len(source_files)
    files_with_errors = 0
    for index_in_source, file_path in enumerate(source_files, start=1):
        if verbose:
            print(f"Parsing {file_path}")
        else:
            rel = file_path.relative_to(project_path)
            sys.stdout.write(f"\rParsing {index_in_source}/{total_files}: {rel}" + " " * 20)
            sys.stdout.flush()
        # Recorded before the parse attempt so a file that fails to parse
        # (e.g. a header clang can't compile standalone) still gets an empty
        # entry in base.json instead of silently vanishing from it entirely.
        processed_files.append(file_path)
        try:
            parse_args = list(clang_args)
            # clang_args defaults to C++ (needed for .cpp/.hpp/... files); C
            # sources and headers must be parsed as C11 instead, or passing a
            # C++-only flag like -std=c++17 to a file libclang treats as a C
            # translation unit (any .c/.h) makes index.parse() raise
            # TranslationUnitLoadError outright instead of just warning, so
            # every function in that file is lost rather than just its LLVM
            # mapping. This previously only special-cased .c, silently
            # dropping every function defined in a .h file (e.g. static
            # inline helpers) across the whole project.
            if file_path.suffix.lower() not in {'.cpp', '.cc', '.cxx', '.hpp', '.hh', '.hxx'}:
                parse_args[0] = '-std=c11'
            tu = index.parse(str(file_path), args=parse_args)

            if tu.diagnostics:
                errors = [d for d in tu.diagnostics if d.severity >= clang.cindex.Diagnostic.Error]
                if errors:
                    files_with_errors += 1
                if verbose:
                    for diag in tu.diagnostics:
                        print(f"Diagnostic: {diag}", file=sys.stderr)

            # Collect functions from this translation unit
            file_functions = []

            def visit_cursor(cursor: clang.cindex.Cursor, parent: Optional[clang.cindex.Cursor] = None):
                if cursor.kind in CLASS_LIKE_KINDS and cursor.is_definition():
                    if _is_project_owned(cursor, str(project_path)):
                        class_id = _class_id_for_cursor(cursor, str(project_path))
                        if class_id and class_id not in class_cursor_map:
                            class_cursor_map[class_id] = cursor
                if cursor.kind in FUNCTION_LIKE_KINDS:
                    # Only consider definitions, not just declarations
                    if cursor.is_definition():
                        if _is_project_owned(cursor, str(project_path)):
                            func_info = FunctionInfo(
                                cursor, str(file_path), str(project_path),
                                include_dirs, compile_cache
                            )
                            if func_info.id not in function_map:
                                file_functions.append(func_info)
                                all_functions.append(func_info)
                                function_map[func_info.id] = func_info
                        elif cursor.location.file:
                            excluded_system_functions.add((
                                os.path.abspath(os.path.realpath(cursor.location.file.name)),
                                cursor.location.line,
                                cursor.spelling
                            ))
                # Recurse
                for child in cursor.get_children():
                    visit_cursor(child, cursor)

            visit_cursor(tu.cursor)
            compiled = _compile_translation_unit(file_path, include_dirs, compile_cache)
            for func in file_functions:
                func.enrich_with_llvm(compiled["ir"], compiled["status"], compiled["error"])
                functions_by_file.setdefault(func.relative_path, []).append(func)
        except Exception as e:
            # Covers both index.parse() itself and everything derived from
            # its translation unit (cursor traversal, LLVM compilation): the
            # libclang bindings decode cursor spellings/diagnostics as strict
            # UTF-8 (clang/cindex.py's c_interop_string.value), so a source
            # file with a non-UTF-8 byte anywhere - even deep inside a macro
            # or comment - can raise UnicodeDecodeError well after parsing
            # succeeded. Treat that the same as a parse failure: skip the
            # file rather than aborting the whole run.
            files_with_errors += 1
            if verbose:
                print(f"\nError parsing {file_path}: {e}", file=sys.stderr)
            continue

        if checkpoint_every > 0 and (
            index_in_source % checkpoint_every == 0 or index_in_source == total_files
        ):
            _finalize_and_write(
                output_file, project_path, processed_files,
                all_functions, function_map, functions_by_file, class_cursor_map,
            )

    # Always finish with one last checkpoint so the file on disk reflects
    # the complete, fully call-resolved result even if checkpoint_every
    # didn't line up with total_files (or checkpointing was disabled).
    _finalize_and_write(
        output_file, project_path, processed_files,
        all_functions, function_map, functions_by_file, class_cursor_map,
    )

    successful_compilations = sum(result["status"] == "compiled" for result in compile_cache.values())
    failed_compilations = sum(
        result["status"] == "compilation_failed" for result in compile_cache.values()
    )
    basic_blocks = sum(len(func.basic_blocks) for func in all_functions)
    cfg_edges = sum(len(func.cfg["edges"]) for func in all_functions)
    matched_functions = sum(func.analysis_status == "success" for func in all_functions)
    matching_failures = sum(
        func.llvm_status == "compiled" and func.analysis_status != "success"
        for func in all_functions
    )
    methods = sum(func.class_info["is_method"] for func in all_functions)
    constructors = sum(func.class_info["is_constructor"] for func in all_functions)
    destructors = sum(func.class_info["is_destructor"] for func in all_functions)
    templates = sum(func.class_info["is_template"] for func in all_functions)
    virtuals = sum(func.class_info["is_virtual"] for func in all_functions)
    statics = sum(func.class_info["is_static"] for func in all_functions)
    operators = sum(func.class_info["is_operator_overload"] for func in all_functions)
    if not verbose:
        sys.stdout.write("\n")
    print(f"Source files discovered: {len(source_files)}")
    if not verbose:
        print(f"Files with parse errors: {files_with_errors} (pass verbose=True / --verbose for details)")
    print(f"Classes/structs discovered: {len(class_cursor_map)}")
    print(f"Project functions: {len(all_functions)}")
    print(f"  of which methods: {methods} (constructors: {constructors}, destructors: {destructors}, "
          f"static: {statics}, virtual: {virtuals}, operator overloads: {operators})")
    print(f"  of which function templates: {templates}")
    print(f"System functions excluded: {len(excluded_system_functions)}")
    print(f"LLVM functions matched: {matched_functions}")
    print(f"Functions with source ranges: {sum(bool(func.source_location['start_line'] and func.source_location['end_line']) for func in all_functions)}")
    print(f"Project headers analyzed: {sum(f.suffix.lower() in {'.h', '.hpp', '.hh', '.hxx'} for f in processed_files)}")
    print(f"Functions with CFG: {matched_functions}")
    print(f"Basic blocks: {basic_blocks}")
    print(f"CFG edges: {cfg_edges}")
    print(f"Compilation failures: {failed_compilations}")
    print(f"LLVM matching failures: {matching_failures}")
    print(f"Output: {output_file}")

    return output_file


def _extract_calls(
    cursor: clang.cindex.Cursor,
    function_map: Dict[str, FunctionInfo],
    project_path: str,
    include_dirs: Optional[List[str]] = None,
    compile_cache: Optional[Dict[str, Dict[str, str]]] = None
) -> List[Dict[str, Any]]:
    """Extract function calls from the given cursor."""
    calls = []
    for child in cursor.get_children():
        if child.kind == CursorKind.CALL_EXPR:
            # Get the called function
            called_ref = child.get_definition()
            if called_ref and called_ref.kind in FUNCTION_LIKE_KINDS:
                if called_ref.is_definition():
                    if not _is_project_owned(called_ref, project_path):
                        continue
                    called_file = os.path.relpath(
                        os.path.abspath(os.path.realpath(called_ref.location.file.name)),
                        os.path.abspath(os.path.realpath(project_path))
                    )
                    called_qualified = _qualified_cursor_name(called_ref)
                    called_id_prefix = f"{called_file}::{called_qualified}::"
                    target = next(
                        (func for func_id, func in function_map.items()
                         if func_id.startswith(called_id_prefix)
                         and func.source_location["start_line"] == called_ref.location.line),
                        None
                    )
                    call = {
                        "name": called_ref.spelling,
                        "qualified_name": called_qualified,
                        "resolved": target is not None,
                    }
                    if target is not None:
                        call["id"] = target.id
                        if target.mangled_name:
                            call["llvm_name"] = target.mangled_name
                    calls.append(call)
        # Recurse
        calls.extend(_extract_calls(
            child, function_map, project_path, include_dirs, compile_cache
        ))
    return calls


def _qualified_cursor_name(cursor: clang.cindex.Cursor) -> str:
    name_parts = []
    while cursor:
        if cursor.kind in NAME_CONTRIBUTING_KINDS and cursor.spelling:
            name_parts.insert(0, cursor.spelling)
        cursor = cursor.semantic_parent
    return "::".join(name_parts)


def _class_cursor_to_dict(
    cursor: clang.cindex.Cursor, class_id: str, project_root: str, method_ids: List[str]
) -> Dict[str, Any]:
    """Build the base.json representation of one extracted class/struct."""
    relative_path = os.path.relpath(
        os.path.abspath(os.path.realpath(cursor.location.file.name)),
        os.path.abspath(os.path.realpath(project_root)),
    )
    base_classes = [
        child.type.spelling for child in cursor.get_children()
        if child.kind == CursorKind.CXX_BASE_SPECIFIER
    ]
    return {
        "id": class_id,
        "name": cursor.spelling,
        "qualified_name": _qualified_cursor_name(cursor),
        "kind": cursor.kind.name,
        "is_abstract": cursor.is_abstract_record(),
        "base_classes": base_classes,
        "source_location": {
            "file": relative_path,
            "start_line": cursor.location.line,
            "end_line": cursor.extent.end.line,
        },
        "method_ids": method_ids,
    }


def _extract_dependencies(cursor: clang.cindex.Cursor, project_path: Path) -> List[str]:
    """Extract dependencies (simplified)."""
    deps = []
    # We'll just look for references to other functions, classes, etc.
    # For now, return an empty list to keep it simple.
    return deps


if __name__ == "__main__":
    # For testing
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analyzer.py <project_path>")
        sys.exit(1)
    project_path = Path(sys.argv[1])
    output_dir = Path("output")
    analyze_project(project_path, output_dir)
    print(f"Analysis complete. Output in {output_dir}")