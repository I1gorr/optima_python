"""base.json download/versioning, run context, and benchmark freezing.

This module owns the single point where base.json is resolved: once per
run, hashed, snapshotted immutably under
``runs/<dataset>/base/base.json`` (``load_base_snapshot``, the
dataset/test-suite-aware entry point -- accepts either a local file path,
e.g. a Kaggle input dataset, or an http(s) URL, and derives ``<dataset>``
from the base.json's own ``project.test_suite``/``project.name`` field so
different test suites never collide), and validated. ``download_base_snapshot``
remains the lower-level, URL-only, purely content-addressed
(``runs/base-<sha12>/base/base.json``) primitive it wraps. Everything
downstream (enrichment, embedding, evaluation) reads the resolved snapshot
path and never re-downloads or regenerates it.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .errors import BaseJsonError, BenchmarkError, ExperimentIncompleteError

_LFS_POINTER_PREFIX = b"version https://git-lfs"


def safe_slug(name: str) -> str:
    """Stable, filesystem-safe identifier for a dataset/test-suite name.
    Matches the slug convention Optima already uses for model/embedding
    identifiers (``optima.rag.embedding_simple._slug``,
    ``optima.analyzer.safe_slug``).
    """
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(name).strip()).strip("-").lower()
    return slug or "dataset"


def _is_url(value: str) -> bool:
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", str(value)))


def _dataset_slug_from_base_json(data: Any, fallback: str) -> str:
    """Prefer the analyzer's own ``project.test_suite`` (or ``project.name``)
    field, embedded in the base.json content itself, over anything derived
    from where the file happens to be mounted/downloaded -- so the same
    dataset is recognized consistently even if BASE_JSON is copied or renamed.
    """
    project = data.get("project") if isinstance(data, dict) else None
    if isinstance(project, dict):
        candidate = project.get("test_suite") or project.get("name")
        if candidate:
            return safe_slug(str(candidate))
    return safe_slug(fallback)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class BaseSnapshot:
    path: Path
    sha256: str
    bytes: int
    url: str
    etag: Optional[str]
    downloaded_at: str
    run_dir: Path
    # Dataset/test-suite slug this snapshot belongs to. Defaults to None
    # (not ``"unknown"``) so existing callers that construct a BaseSnapshot
    # directly (e.g. tests) without a dataset keep working unchanged.
    dataset: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        data["run_dir"] = str(self.run_dir)
        return data


def _download_stream(url: str, dest: Path, timeout: float = 60.0, attempts: int = 3) -> tuple[str, int, Optional[str]]:
    """Stream url to dest, retrying with backoff. Returns (sha256, bytes, etag)."""
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "optima-kaggle/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                etag = response.headers.get("ETag")
                hasher = hashlib.sha256()
                total = 0
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("wb") as out:
                    first_chunk = True
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        if first_chunk:
                            if chunk.startswith(_LFS_POINTER_PREFIX):
                                raise BaseJsonError(
                                    f"{url} resolved to a Git LFS pointer file, not the "
                                    f"actual content. Use the LFS media URL "
                                    f"(media.githubusercontent.com/media/...) or attach "
                                    f"base.json as a GitHub Release asset instead."
                                )
                            first_chunk = False
                        hasher.update(chunk)
                        total += len(chunk)
                        out.write(chunk)
                return hasher.hexdigest(), total, etag
        except BaseJsonError:
            raise
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(2 ** attempt)
                continue
    raise BaseJsonError(f"Failed to download {url} after {attempts} attempts: {last_error}")


def download_base_snapshot(
    url: str,
    output_root: Path,
    expected_sha256: Optional[str] = None,
    timeout: float = 60.0,
    attempts: int = 3,
) -> BaseSnapshot:
    """Download base.json (or base.json.gz) once, snapshot it immutably, and
    return the snapshot descriptor. Safe to call repeatedly: a matching
    snapshot already on disk is verified and reused without re-downloading.
    """
    output_root = Path(output_root)
    downloads_dir = output_root / "downloads"
    part_path = downloads_dir / f"base-{uuid.uuid4().hex[:8]}.part"
    try:
        sha256, num_bytes, etag = _download_stream(url, part_path, timeout=timeout, attempts=attempts)
        is_gzip = url.endswith(".gz")
        compressed_sha256 = sha256
        if is_gzip:
            decompressed_path = part_path.with_suffix(".json")
            try:
                with gzip.open(part_path, "rb") as src, decompressed_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            except OSError as exc:
                raise BaseJsonError(f"{url} looks gzip-compressed but could not be decompressed: {exc}")
            sha256 = _sha256_file(decompressed_path)
            num_bytes = decompressed_path.stat().st_size
            part_path.unlink(missing_ok=True)
            part_path = decompressed_path

        if expected_sha256 and sha256 != expected_sha256:
            raise BaseJsonError(
                f"base.json sha256 mismatch: expected {expected_sha256}, got {sha256}. "
                f"The file at {url} does not match BASE_JSON_EXPECTED_SHA256."
            )

        sha12 = sha256[:12]
        run_dir = output_root / "runs" / f"base-{sha12}"
        base_dir = run_dir / "base"
        base_path = base_dir / "base.json"
        base_dir.mkdir(parents=True, exist_ok=True)

        if base_path.exists():
            existing_sha = _sha256_file(base_path)
            if existing_sha != sha256:
                raise BaseJsonError(
                    f"{base_path} already exists with a different hash "
                    f"({existing_sha[:12]} on disk vs {sha12} just downloaded). "
                    f"This should be impossible for a content-addressed run "
                    f"directory; remove {run_dir} if it is stale."
                )
            part_path.unlink(missing_ok=True)
        else:
            part_path.replace(base_path)

        snapshot = BaseSnapshot(
            path=base_path, sha256=sha256, bytes=num_bytes, url=url, etag=etag,
            downloaded_at=_now_iso(), run_dir=run_dir,
        )
        meta = snapshot.as_dict()
        meta["compressed_sha256"] = compressed_sha256 if is_gzip else None
        (base_dir / "base_snapshot.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        print(f"BASE SNAPSHOT: base-{sha12} | {num_bytes:,} bytes | {url}")
        return snapshot
    finally:
        if part_path.exists() and part_path.name.endswith(".part"):
            part_path.unlink(missing_ok=True)


def load_base_snapshot(
    base_json: str,
    output_root: Path,
    expected_sha256: Optional[str] = None,
    dataset: Optional[str] = None,
    timeout: float = 60.0,
    attempts: int = 3,
) -> BaseSnapshot:
    """Resolve ``BASE_JSON`` -- a local file path (e.g. a Kaggle input
    dataset mount) or an http(s) URL -- into an immutable, dataset-addressed
    snapshot at ``runs/<dataset>/base/base.json``. Safe to call repeatedly:
    a matching snapshot already on disk is verified and reused.

    Selecting a different dataset/model combination is entirely driven by
    what ``base_json`` and ``dataset``/the caller's model choice are -- this
    function never looks at or depends on which model will be used.

    ``dataset`` overrides dataset detection when given; otherwise the
    dataset slug comes from the base.json content's own
    ``project.test_suite`` (or ``project.name``) field so the same dataset
    is recognized consistently regardless of the path/URL it was loaded
    from, falling back to the source path/URL's own directory name only
    when the content has neither field (e.g. a pre-existing base.json
    generated before this field was added).

    A local path is only ever read and copied, never moved or written to --
    safe for a read-only Kaggle input mount.
    """
    output_root = Path(output_root)
    value = str(base_json)

    if _is_url(value):
        downloads_dir = output_root / "downloads"
        part_path = downloads_dir / f"base-{uuid.uuid4().hex[:8]}.part"
        try:
            sha256, num_bytes, etag = _download_stream(value, part_path, timeout=timeout, attempts=attempts)
            is_gzip = value.endswith(".gz")
            if is_gzip:
                decompressed_path = part_path.with_suffix(".json")
                try:
                    with gzip.open(part_path, "rb") as src, decompressed_path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                except OSError as exc:
                    raise BaseJsonError(f"{value} looks gzip-compressed but could not be decompressed: {exc}")
                sha256 = _sha256_file(decompressed_path)
                num_bytes = decompressed_path.stat().st_size
                part_path.unlink(missing_ok=True)
                part_path = decompressed_path
            source_path = part_path
            source_label = value
            fallback_name = Path(value.split("?")[0].rstrip("/")).parent.name or "dataset"
            movable = True
        except BaseException:
            if part_path.exists():
                part_path.unlink(missing_ok=True)
            raise
    else:
        source_path = Path(value).expanduser()
        if not source_path.is_absolute():
            source_path = (Path.cwd() / source_path).resolve()
        else:
            source_path = source_path.resolve()
        if not source_path.exists():
            raise BaseJsonError(
                f"BASE_JSON={value!r} is neither a URL nor an existing local file."
            )
        if not source_path.is_file():
            raise BaseJsonError(f"BASE_JSON={value!r} is not a file.")
        sha256 = _sha256_file(source_path)
        num_bytes = source_path.stat().st_size
        etag = None
        source_label = str(source_path)
        fallback_name = source_path.parent.name or "dataset"
        movable = False

    if expected_sha256 and sha256 != expected_sha256:
        raise BaseJsonError(
            f"base.json sha256 mismatch: expected {expected_sha256}, got {sha256}. "
            f"{source_label} does not match BASE_JSON_EXPECTED_SHA256."
        )

    try:
        content = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        if movable:
            part_path.unlink(missing_ok=True)
        raise BaseJsonError(f"{source_label} could not be parsed as JSON: {exc}") from exc

    dataset_slug = safe_slug(dataset) if dataset else _dataset_slug_from_base_json(content, fallback_name)

    # Dataset-addressed, human-readable path -- NOT content-hash-addressed --
    # so runs/<dataset>/... stays predictable across re-runs (matching the
    # local analyzer's optima_outputs/base/<test-suite>/base.json layout).
    # sha256 is still recorded (base_snapshot.json, the run manifest, and
    # every enhanced_<model>.json) as the fingerprint that proves exactly
    # which base.json content produced a given result.
    run_dir = output_root / "runs" / dataset_slug
    base_dir = run_dir / "base"
    base_path = base_dir / "base.json"
    base_dir.mkdir(parents=True, exist_ok=True)

    if base_path.exists():
        existing_sha = _sha256_file(base_path)
        if existing_sha != sha256:
            raise BaseJsonError(
                f"{base_path} already holds a different base.json for dataset "
                f"{dataset_slug!r} (sha256 {existing_sha[:12]} on disk vs "
                f"{sha256[:12]} from {source_label}). Each dataset name maps to "
                f"one base.json; remove {base_path} if it was intentionally "
                f"regenerated, or pass a distinct `dataset` name/BASE_JSON "
                f"test_suite to keep both."
            )
        if movable:
            part_path.unlink(missing_ok=True)
    elif movable:
        part_path.replace(base_path)
    else:
        shutil.copy2(source_path, base_path)

    snapshot = BaseSnapshot(
        path=base_path, sha256=sha256, bytes=num_bytes, url=source_label, etag=etag,
        downloaded_at=_now_iso(), run_dir=run_dir, dataset=dataset_slug,
    )
    meta = snapshot.as_dict()
    (base_dir / "base_snapshot.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"BASE SNAPSHOT: dataset={dataset_slug} | sha256={sha256[:12]} | {num_bytes:,} bytes | {source_label}")
    return snapshot


def validate_base_snapshot(path: Path, expected_function_count: Optional[int] = None) -> dict[str, Any]:
    """Validate an Optima base.json artifact and write a diagnostics report
    next to it. Raises BaseJsonError on any hard-error condition.
    """
    # Imported lazily: colab_pipeline lives in the cloned Optima repo, which
    # is only on sys.path once the notebook's clone cell has run.
    from colab.colab_pipeline import flatten_functions, load_json, validate_json

    path = Path(path)
    if not path.exists():
        raise BaseJsonError(f"base.json snapshot not found at {path}.")

    try:
        info = validate_json(path)
    except (ValueError, json.JSONDecodeError) as exc:
        raise BaseJsonError(f"{path} failed Optima JSON validation: {exc}") from exc

    if not info["valid"]:
        raise BaseJsonError(
            f"{path} failed Optima schema validation: "
            f"{len(info['malformed_files'])} malformed files, "
            f"{len(info['malformed_nodes'])} malformed function nodes. "
            f"First few: {info['malformed_nodes'][:5]}"
        )
    if info["status"] != "BASE JSON":
        raise BaseJsonError(
            f"{path} is already an ENRICHED JSON artifact (has enrichment fields: "
            f"{info['enrichment_fields']}). This experiment must start from a "
            f"base.json with no enrichment; point BASE_JSON_URL at the raw "
            f"analyzer output instead."
        )

    data = load_json(path)
    functions = flatten_functions(data)
    if not isinstance(data.get("project"), dict):
        raise BaseJsonError(f"{path}: 'project' is missing or not an object.")
    if not data.get("files"):
        raise BaseJsonError(f"{path}: 'files' is empty.")
    if not functions:
        raise BaseJsonError(f"{path}: no functions found under files[].functions[].")

    ids = [function.get("id") for function in functions]
    duplicate_ids = sorted({fid for fid in ids if ids.count(fid) > 1})
    if duplicate_ids:
        raise BaseJsonError(
            f"{path}: duplicate function IDs found ({len(duplicate_ids)}); "
            f"resume is keyed by ID, so this is a hard error. "
            f"Examples: {duplicate_ids[:5]}"
        )

    if expected_function_count is not None and len(functions) != expected_function_count:
        raise BaseJsonError(
            f"{path}: expected {expected_function_count} functions, found "
            f"{len(functions)}. Set EXPECTED_FUNCTION_COUNT=None to disable "
            f"this check if the base.json is intentionally different."
        )

    status_counts: dict[str, int] = {}
    for function in functions:
        status = function.get("analysis_status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    presence_fields = (
        "return_type", "parameters", "source_location", "ast", "cfg",
        "basic_blocks", "calls", "called_by", "dependencies", "llvm",
        "llvm_ir", "mangled_name", "compiler_error",
    )
    presence_counts = {
        field_name: sum(1 for f in functions if f.get(field_name)) for field_name in presence_fields
    }
    empty_source_count = sum(1 for f in functions if not f.get("source_code"))

    report = {
        "path": str(path),
        "valid": True,
        "status": info["status"],
        "files": info["files"],
        "functions": len(functions),
        "unique_function_ids": len(set(ids)),
        "analysis_status_counts": status_counts,
        "presence_counts": presence_counts,
        "empty_source_code_count": empty_source_count,
        "checked_at": _now_iso(),
    }
    (path.parent / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"base.json validated: {report['files']} files, {report['functions']} functions, "
          f"analysis_status={status_counts}")
    if empty_source_count:
        print(f"  warning: {empty_source_count} functions have no source_code")
    return report


@dataclass
class RunContext:
    """All filesystem paths and shared metadata for one base.json snapshot run."""

    run_dir: Path
    base: BaseSnapshot
    config: dict[str, Any]
    env: dict[str, Any]
    optima_commit: str
    session_id: str
    configured_models: list[str] = field(default_factory=list)
    manifest_path: Path = field(init=False)
    gpu_log_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.gpu_log_path = self.run_dir / "logs" / "gpu_memory.jsonl"

    @property
    def base_dir(self) -> Path:
        return self.run_dir / "base"

    @property
    def benchmark_dir(self) -> Path:
        return self.run_dir / "benchmark"

    @property
    def smoke_dir(self) -> Path:
        return self.run_dir / "smoke"

    @property
    def enriched_dir(self) -> Path:
        return self.run_dir / "enriched"

    @property
    def rag_inputs_dir(self) -> Path:
        return self.run_dir / "rag_inputs"

    @property
    def rag_dir(self) -> Path:
        return self.run_dir / "rag"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def summaries_dir(self) -> Path:
        return self.run_dir / "summaries"

    @property
    def dataset(self) -> str:
        return self.base.dataset or "unknown"

    def load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {}

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")

    def set_model_status(self, slug: str, status: str, details: Optional[dict[str, Any]] = None) -> None:
        manifest = self.load_manifest()
        models = manifest.setdefault("models", {})
        models[slug] = {"status": status, "updated_at": _now_iso(), **(details or {})}
        self.save_manifest(manifest)

    def get_model_status(self, slug: str) -> Optional[str]:
        return self.load_manifest().get("models", {}).get(slug, {}).get("status")

    @classmethod
    def create(
        cls,
        output_root: Path,
        base: BaseSnapshot,
        config: dict[str, Any],
        env: dict[str, Any],
        optima_commit: str,
        configured_models: Optional[list[str]] = None,
    ) -> "RunContext":
        ctx = cls(
            run_dir=base.run_dir, base=base, config=config, env=env,
            optima_commit=optima_commit, session_id=uuid.uuid4().hex,
            configured_models=configured_models or [],
        )
        for sub in (ctx.base_dir, ctx.benchmark_dir, ctx.smoke_dir, ctx.enriched_dir,
                    ctx.rag_inputs_dir, ctx.rag_dir, ctx.logs_dir, ctx.summaries_dir):
            sub.mkdir(parents=True, exist_ok=True)
        manifest = ctx.load_manifest()
        manifest.setdefault("dataset", base.dataset)
        manifest.setdefault("base_sha256", base.sha256)
        manifest.setdefault("base_json", base.url)
        manifest.setdefault("base_url", base.url)  # kept for backward compatibility
        manifest.setdefault("created_at", _now_iso())
        manifest["optima_commit"] = optima_commit
        manifest["last_session_id"] = ctx.session_id
        manifest["config"] = config
        manifest["env"] = env
        manifest["output_dir"] = str(ctx.run_dir)
        manifest["configured_models"] = sorted(set(manifest.get("configured_models", [])) | set(ctx.configured_models))
        manifest.setdefault("models", {})
        ctx.save_manifest(manifest)
        return ctx


def restore_resume_state(ctx: RunContext, resume_input_dir: Optional[Path]) -> None:
    """Copy prior checkpoint/smoke/log artifacts for this same base sha from a
    previous run's output (e.g. attached as a Kaggle input dataset). Never
    overwrites a local file with a shorter one.
    """
    if resume_input_dir is None:
        return
    resume_input_dir = Path(resume_input_dir)
    sha12 = ctx.base.sha256[:12]
    source_run_dir = resume_input_dir / "runs" / f"base-{sha12}"
    if not source_run_dir.exists():
        print(f"RESUME: no matching run directory for base-{sha12} under {resume_input_dir}; nothing restored.")
        return

    def _line_count(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("rb") as handle:
            return sum(1 for _ in handle)

    copied = []
    for pattern in ("enriched/*.checkpoint.jsonl", "smoke/**/*.jsonl", "smoke/**/*.json",
                    "logs/*/attempts.jsonl"):
        for src in source_run_dir.glob(pattern):
            rel = src.relative_to(source_run_dir)
            dst = ctx.run_dir / rel
            if src.suffix == ".jsonl" and _line_count(src) <= _line_count(dst):
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(str(rel))
    print(f"RESUME: restored {len(copied)} file(s) from {source_run_dir}")
    for rel in copied:
        print(f"  {rel}")


def freeze_benchmark(
    ctx: RunContext,
    num_queries: int = 20,
    seed: int = 42,
    benchmark_json_url: Optional[str] = None,
) -> dict[str, Any]:
    """Generate (or download) the benchmark exactly once for this base.json
    snapshot, freeze it to disk, and validate it. Never regenerated from a
    directory that also contains enriched JSON files (that would let each
    enrichment model grade its own benchmark).
    """
    from colab.colab_pipeline import flatten_functions, load_json
    from optima.rag.benchmark_generator import validate_benchmark_queries
    from optima.rag.evaluation import (
        create_benchmark_from_json_files,
        load_benchmark_queries,
        save_benchmark_queries,
    )

    queries_path = ctx.benchmark_dir / "queries.json"
    manifest_path = ctx.benchmark_dir / "benchmark_manifest.json"

    if queries_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("base_sha256") != ctx.base.sha256:
            raise BenchmarkError(
                f"{manifest_path} was generated for base_sha256="
                f"{manifest.get('base_sha256')}, but the current snapshot is "
                f"{ctx.base.sha256}. This should not happen inside one run "
                f"directory; delete {ctx.benchmark_dir} if it is stale."
            )
        queries = load_benchmark_queries(queries_path)
        print(f"BENCHMARK: reusing frozen benchmark ({len(queries)} queries) from {queries_path}")
        return {"path": queries_path, "manifest": manifest, "queries": queries}

    if benchmark_json_url:
        raw_path = ctx.benchmark_dir / "downloaded_queries.json"
        sha256, num_bytes, _etag = _download_stream(benchmark_json_url, raw_path)
        queries = json.loads(raw_path.read_text(encoding="utf-8"))
        source = benchmark_json_url
    else:
        # base_dir contains ONLY base.json (plus its own metadata files),
        # never an enhanced_*.json -- this is what prevents benchmark
        # contamination by a model's own enrichment summaries.
        queries = create_benchmark_from_json_files(ctx.base_dir, num_queries, seed=seed)
        source = "base_only"

    if not queries:
        raise BenchmarkError(
            f"Benchmark generation produced 0 queries from {ctx.base_dir}. "
            f"Check that base.json has usable source_code/enrichment evidence."
        )

    base_data = load_json(ctx.base.path)
    functions = flatten_functions(base_data)
    validation = validate_benchmark_queries(queries, functions)
    min_required = max(10, int(0.5 * num_queries))
    if (validation.get("invalid_function_ids", 0) or validation.get("invalid_schema", 0)
            or validation.get("empty_relevant_functions", 0)):
        raise BenchmarkError(f"Benchmark failed validation: {validation}")
    if len(queries) < min_required:
        raise BenchmarkError(
            f"Only {len(queries)} benchmark queries were generated; need at "
            f"least {min_required} (50% of requested {num_queries})."
        )

    save_benchmark_queries(queries, queries_path)
    manifest = {
        "base_sha256": ctx.base.sha256,
        "sha256": _sha256_file(queries_path),
        "seed": seed,
        "num_queries_requested": num_queries,
        "num_queries_generated": len(queries),
        "source": source,
        "validation": validation,
        "created_at": _now_iso(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"BENCHMARK: generated and froze {len(queries)} queries -> {queries_path}")
    return {"path": queries_path, "manifest": manifest, "queries": queries}


def _format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def final_report(ctx: RunContext) -> dict[str, Any]:
    """Print the artifact tree with sizes and raise if any configured model
    did not reach a passing state.
    """
    print(f"\nArtifact tree for {ctx.run_dir}:")
    total_bytes = 0
    for path in sorted(ctx.run_dir.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            total_bytes += size
            print(f"  {path.relative_to(ctx.run_dir)}  ({_format_size(size)})")
    print(f"Total: {_format_size(total_bytes)}")
    print(f"Manifest: {ctx.manifest_path}")

    manifest = ctx.load_manifest()
    failed = []
    for slug in manifest.get("configured_models", []):
        status = manifest.get("models", {}).get(slug, {}).get("status")
        if status not in ("enrichment_passed",):
            failed.append((slug, status))
    if failed:
        raise ExperimentIncompleteError(
            f"{len(failed)} configured model(s) did not reach 'enrichment_passed': "
            f"{failed}"
        )
    return {"total_bytes": total_bytes, "manifest": manifest}
