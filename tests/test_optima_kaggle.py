"""Unit tests for optima_kaggle.

These run on CPU without a Kaggle GPU. A few tests use the machine's real
GPU (if present) or the network (Hugging Face config downloads only, never
weights) to sanity-check the meta-device fit-estimation logic; everything
else uses fakes/monkeypatches so the suite stays fast and deterministic.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import unittest
from pathlib import Path
from unittest import mock

from optima_kaggle import enrichment, environment, models, retrieval_eval, snapshot
from optima_kaggle.enrichment import Checkpoint, GenerationSettings
from optima_kaggle.errors import (
    BaseJsonError,
    CheckpointCorruptError,
    ComparisonMismatchError,
    EmbeddingError,
    GateFailedError,
    GenerationOOMError,
    ModelDoesNotFitError,
    ModelNotFeasibleError,
    NoCorporaError,
    PromptTooLongError,
    ResumeMismatchError,
)


ARTIFACTS_ROOT = Path(__file__).parent / ".optima_kaggle_artifacts"
REAL_BASE_JSON = Path(__file__).parent.parent / "output" / "base.json"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _minimal_base_json(function_count: int = 3, unique_ids: bool = True) -> dict:
    functions = []
    for i in range(function_count):
        fid = f"file.cpp::fn{i if unique_ids else 0}::{i if unique_ids else 0}"
        functions.append({
            "id": fid, "name": f"fn{i}", "qualified_name": f"ns::fn{i}",
            "return_type": "int", "parameters": [],
            "source_code": f"int fn{i}(int x) {{ return x + {i}; }}",
            "analysis_status": "success",
            "source_location": {"file": "file.cpp", "start_line": i * 10},
            "ast": "{}", "cfg": {}, "calls": [], "called_by": [], "dependencies": [],
            "llvm": {}, "llvm_ir": "", "mangled_name": f"_Zfn{i}",
        })
    return {
        "project": {"name": "test", "root": "/tmp/test", "language": "cpp"},
        "files": [{"path": "file.cpp", "name": "file.cpp", "relative_path": "file.cpp",
                   "id": "file.cpp", "language": "cpp", "functions": functions}],
    }


class FakeTokenizer:
    """Minimal stand-in for a HF tokenizer: one 'token' per whitespace-split word."""

    pad_token_id = 0
    eos_token_id = 1
    chat_template = "fake-template"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def __call__(self, text, add_special_tokens=True, return_tensors=None, **kwargs):
        ids = text.split()
        return {"input_ids": ids}


def _fake_spec(max_input_tokens=50, max_new_tokens=32, quantization="fp16",
                single_gpu_tier="A", slug="fake-model", allow_cpu_offload=False):
    return models.ModelSpec(
        slug=slug, model_id="fake-org/fake-model", quantization=quantization,
        single_gpu_tier=single_gpu_tier, allow_cpu_offload=allow_cpu_offload,
        max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens,
    )


class _FakeHandle:
    def __init__(self, spec, tokenizer=None):
        self.spec = spec
        self.tokenizer = tokenizer or FakeTokenizer()
        self.model = object()
        self.input_device = "cpu"
        self.load_report = {"footprint_gib": 1.0}


def _gen_result(text, finish_reason="eos", input_tokens=10, output_tokens=5,
                 latency_s=0.05, peak=0.5, free=5.0):
    return models.GenerationResult(
        text=text, input_tokens=input_tokens, output_tokens=output_tokens,
        latency_s=latency_s, finish_reason=finish_reason,
        peak_allocated_gib=peak, free_after_gib=free,
    )


VALID_ENRICHMENT_JSON = json.dumps({
    "purpose": "Adds an offset to x.", "behavior": "Returns x plus a constant offset.",
    "summary": "Simple additive function.", "inputs": ["x"], "outputs": ["sum"],
    "side_effects": [], "dependencies": [], "concepts": ["arithmetic"], "keywords": ["add"],
    "algorithm": "Direct addition.", "complexity": {"time": "O(1)", "space": "O(1)"},
})


class EnvironmentTests(unittest.TestCase):
    def test_parse_version_orders_correctly(self):
        self.assertLess(environment._parse_version("0.42.0"), environment._parse_version("0.43.0"))
        self.assertGreater(environment._parse_version("0.45.1"), environment._parse_version("0.45.0"))

    def test_ensure_python_deps_reports_installed_packages_without_installing(self):
        report = environment.ensure_python_deps(allow_install=False)
        self.assertIn("langchain_core", report["found"])
        self.assertEqual(report["installed_now"], [])

    def test_probe_bitsandbytes_reports_not_installed_without_side_effects(self):
        status = environment.probe_bitsandbytes(allow_install=False)
        self.assertFalse(status.ok)
        self.assertEqual(status.stage, "not_installed")

    def test_assert_gpu_clean_passes_when_idle(self):
        try:
            import torch
            if not torch.cuda.is_available():
                self.skipTest("no CUDA device available in this environment")
        except ImportError:
            self.skipTest("torch not installed")
        environment.assert_gpu_clean(threshold_gib=0.5)  # should not raise


class SnapshotDownloadTests(unittest.TestCase):
    root = ARTIFACTS_ROOT / "snapshot_download"

    def setUp(self):
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_source(self, name: str, content: bytes) -> str:
        path = self.root / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path.resolve().as_uri()

    def test_download_creates_content_addressed_snapshot(self):
        content = json.dumps(_minimal_base_json()).encode("utf-8")
        url = self._write_source("base.json", content)
        output_root = self.root / "outputs"
        snap = snapshot.download_base_snapshot(url, output_root)
        self.assertTrue(snap.path.exists())
        self.assertEqual(snap.sha256, hashlib.sha256(content).hexdigest())
        self.assertEqual(snap.run_dir.name, f"base-{snap.sha256[:12]}")
        self.assertTrue((snap.run_dir / "base" / "base_snapshot.json").exists())

    def test_download_is_idempotent_for_same_content(self):
        content = json.dumps(_minimal_base_json()).encode("utf-8")
        url = self._write_source("base.json", content)
        output_root = self.root / "outputs"
        first = snapshot.download_base_snapshot(url, output_root)
        second = snapshot.download_base_snapshot(url, output_root)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.path, second.path)

    def test_expected_sha256_mismatch_raises(self):
        content = json.dumps(_minimal_base_json()).encode("utf-8")
        url = self._write_source("base.json", content)
        with self.assertRaises(BaseJsonError):
            snapshot.download_base_snapshot(url, self.root / "outputs", expected_sha256="0" * 64)

    def test_git_lfs_pointer_is_rejected(self):
        pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\nsize 123\n"
        url = self._write_source("base.json", pointer)
        with self.assertRaises(BaseJsonError):
            snapshot.download_base_snapshot(url, self.root / "outputs")

    def test_gzip_source_is_decompressed_and_hashed_on_decompressed_content(self):
        import gzip
        content = json.dumps(_minimal_base_json()).encode("utf-8")
        gz_path = self.root / "source" / "base.json.gz"
        gz_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(gz_path, "wb") as handle:
            handle.write(content)
        url = gz_path.resolve().as_uri() + ".gz" if False else gz_path.resolve().as_uri()
        # as_uri() already ends in .gz since the file is named base.json.gz
        snap = snapshot.download_base_snapshot(url, self.root / "outputs")
        self.assertEqual(snap.sha256, hashlib.sha256(content).hexdigest())
        self.assertEqual(json.loads(snap.path.read_text())["project"]["name"], "test")


class SnapshotValidateTests(unittest.TestCase):
    root = ARTIFACTS_ROOT / "snapshot_validate"

    def setUp(self):
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write(self, data: dict) -> Path:
        path = self.root / "base.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_valid_base_json_passes(self):
        path = self._write(_minimal_base_json(function_count=5))
        report = snapshot.validate_base_snapshot(path)
        self.assertTrue(report["valid"])
        self.assertEqual(report["functions"], 5)
        self.assertTrue((path.parent / "validation_report.json").exists())

    def test_expected_function_count_mismatch_raises(self):
        path = self._write(_minimal_base_json(function_count=5))
        with self.assertRaises(BaseJsonError):
            snapshot.validate_base_snapshot(path, expected_function_count=96)

    def test_duplicate_function_ids_raise(self):
        path = self._write(_minimal_base_json(function_count=3, unique_ids=False))
        with self.assertRaises(BaseJsonError):
            snapshot.validate_base_snapshot(path)

    def test_already_enriched_json_is_rejected(self):
        data = _minimal_base_json(function_count=1)
        data["files"][0]["functions"][0]["enrichment"] = json.loads(VALID_ENRICHMENT_JSON)
        path = self._write(data)
        with self.assertRaises(BaseJsonError):
            snapshot.validate_base_snapshot(path)

    def test_missing_files_key_raises(self):
        path = self._write({"project": {"name": "x"}})
        with self.assertRaises(BaseJsonError):
            snapshot.validate_base_snapshot(path)


class RunContextTests(unittest.TestCase):
    root = ARTIFACTS_ROOT / "run_context"

    def setUp(self):
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _snap(self) -> snapshot.BaseSnapshot:
        content = json.dumps(_minimal_base_json()).encode("utf-8")
        sha = hashlib.sha256(content).hexdigest()
        run_dir = self.root / "runs" / f"base-{sha[:12]}"
        base_path = run_dir / "base" / "base.json"
        base_path.parent.mkdir(parents=True)
        base_path.write_bytes(content)
        return snapshot.BaseSnapshot(path=base_path, sha256=sha, bytes=len(content),
                                     url="file:///fake", etag=None, downloaded_at="now", run_dir=run_dir)

    def test_create_initializes_directories_and_manifest(self):
        ctx = snapshot.RunContext.create(self.root, self._snap(), {"a": 1}, {"gpu": "fake"},
                                         "deadbeef", configured_models=["m1", "m2"])
        for sub in (ctx.base_dir, ctx.benchmark_dir, ctx.smoke_dir, ctx.enriched_dir,
                    ctx.rag_inputs_dir, ctx.rag_dir, ctx.logs_dir, ctx.summaries_dir):
            self.assertTrue(sub.exists())
        manifest = ctx.load_manifest()
        self.assertEqual(manifest["configured_models"], ["m1", "m2"])

    def test_set_and_get_model_status_roundtrip(self):
        ctx = snapshot.RunContext.create(self.root, self._snap(), {}, {}, "deadbeef")
        ctx.set_model_status("m1", "enrichment_passed", {"success_rate": 0.99})
        self.assertEqual(ctx.get_model_status("m1"), "enrichment_passed")
        self.assertIsNone(ctx.get_model_status("unknown"))


@unittest.skipUnless(REAL_BASE_JSON.exists(), "requires the repository's output/base.json fixture")
class FreezeBenchmarkTests(unittest.TestCase):
    root = ARTIFACTS_ROOT / "freeze_benchmark"

    def setUp(self):
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        sha = hashlib.sha256(REAL_BASE_JSON.read_bytes()).hexdigest()
        run_dir = self.root / "runs" / f"base-{sha[:12]}"
        base_dir = run_dir / "base"
        base_dir.mkdir(parents=True)
        # Symlink instead of copying an ~80MB fixture.
        (base_dir / "base.json").symlink_to(REAL_BASE_JSON.resolve())
        self.snap = snapshot.BaseSnapshot(path=base_dir / "base.json", sha256=sha,
                                          bytes=REAL_BASE_JSON.stat().st_size, url="file:///real",
                                          etag=None, downloaded_at="now", run_dir=run_dir)
        self.ctx = snapshot.RunContext.create(self.root, self.snap, {}, {}, "deadbeef")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_freeze_generates_and_caches_benchmark(self):
        bench = snapshot.freeze_benchmark(self.ctx, num_queries=12, seed=42)
        self.assertGreaterEqual(len(bench["queries"]), 10)
        self.assertTrue(bench["path"].exists())
        bench_again = snapshot.freeze_benchmark(self.ctx, num_queries=12, seed=42)
        self.assertEqual(bench["queries"], bench_again["queries"])

    def test_freeze_is_immune_to_enhanced_json_outside_base_dir(self):
        bench_clean = snapshot.freeze_benchmark(self.ctx, num_queries=12, seed=42)
        (self.ctx.benchmark_dir / "queries.json").unlink()
        (self.ctx.benchmark_dir / "benchmark_manifest.json").unlink()

        poison = self.ctx.enriched_dir / "enhanced_poison-model.json"
        poison.write_text(json.dumps({
            "project": {"name": "poison"},
            "files": [{"path": "p.cpp", "functions": [{
                "id": "p.cpp::poisoned::1", "name": "poisoned", "qualified_name": "poisoned",
                "source_code": "void poisoned() {}",
                "enrichment": {**json.loads(VALID_ENRICHMENT_JSON), "model": "poison",
                               "status": "completed"},
            }]}],
        }))
        bench_after_poison = snapshot.freeze_benchmark(self.ctx, num_queries=12, seed=42)
        self.assertEqual(bench_clean["queries"], bench_after_poison["queries"])


class ModelSpecTests(unittest.TestCase):
    def test_slug_cannot_collide_with_reserved_corpus_names(self):
        with self.assertRaises(ValueError):
            models.ModelSpec(slug="raw", model_id="x", quantization="fp16",
                             single_gpu_tier="A", max_input_tokens=10)

    def test_resolve_unknown_slug_raises_key_error(self):
        with self.assertRaises(KeyError):
            models.resolve_spec("not-a-real-model")

    def test_tier_c_is_refused_on_single_gpu_by_default(self):
        with self.assertRaises(ModelNotFeasibleError):
            models.resolve_spec("qwen25-32b-instruct-nf4", num_gpus=1)

    def test_tier_c_can_be_force_resolved_on_single_gpu(self):
        spec = models.resolve_spec("qwen25-32b-instruct-nf4", num_gpus=1, allow_infeasible=True)
        self.assertEqual(spec.single_gpu_tier, "C")

    def test_tier_c_resolves_normally_with_two_gpus(self):
        spec = models.resolve_spec("qwen25-32b-instruct-nf4", num_gpus=2)
        self.assertEqual(spec.single_gpu_tier, "C")

    def test_spec_from_model_id_builds_a_usable_spec_without_the_registry(self):
        spec = models.spec_from_model_id("Qwen/Qwen2.5-Coder-7B-Instruct")
        self.assertEqual(spec.model_id, "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.assertEqual(spec.quantization, "nf4")
        self.assertEqual(spec.max_input_tokens, 1500)
        self.assertEqual(spec.max_new_tokens, 256)
        self.assertNotIn(spec.slug, models.MODEL_REGISTRY)  # confirms no registry involvement
        # slug must be self-consistent with ModelSpec's own validation
        models.ModelSpec(slug=spec.slug, model_id=spec.model_id, quantization=spec.quantization,
                         single_gpu_tier=spec.single_gpu_tier, max_input_tokens=spec.max_input_tokens)

    def test_spec_from_model_id_honors_overrides(self):
        spec = models.spec_from_model_id("org/some-model", quantization="fp16",
                                         max_input_tokens=800, max_new_tokens=128)
        self.assertEqual(spec.quantization, "fp16")
        self.assertEqual(spec.max_input_tokens, 800)
        self.assertEqual(spec.max_new_tokens, 128)


class ModelFitTests(unittest.TestCase):
    def test_estimate_weights_gib_against_real_hf_config(self):
        """Network call for config.json only (no weight download); validates
        the meta-device parameter counting logic against a real model.
        """
        try:
            spec = models.resolve_spec("qwen25-3b-instruct-fp16")
            info = models.estimate_weights_gib(spec)
        except Exception as exc:  # noqa: BLE001 - environment/network may be unavailable
            self.skipTest(f"could not reach Hugging Face Hub: {exc}")
        self.assertGreater(info["quantizable_params"], info["other_params"])
        self.assertTrue(4.0 <= info["weights_gib"] <= 8.0, info["weights_gib"])

    @staticmethod
    def _fake_estimate(weights_gib, config=None):
        config = config or mock.Mock(num_hidden_layers=2, num_key_value_heads=2, hidden_size=64,
                                     num_attention_heads=2, head_dim=32)
        return {"weights_gib": weights_gib, "config": config, "quantizable_params": 1, "other_params": 1}

    def test_check_fit_raises_when_synthetic_free_memory_is_too_small(self):
        spec = _fake_spec()
        with mock.patch.object(models, "estimate_weights_gib", return_value=self._fake_estimate(10.0)), \
             mock.patch("torch.cuda.mem_get_info", return_value=(1 * 1024**3, 8 * 1024**3)):
            with self.assertRaises(ModelDoesNotFitError):
                models.check_fit(spec, num_gpus=1)

    def test_check_fit_chooses_single_gpu_placement_when_one_gpu_suffices(self):
        spec = _fake_spec()
        with mock.patch.object(models, "estimate_weights_gib", return_value=self._fake_estimate(1.0)), \
             mock.patch("torch.cuda.mem_get_info", return_value=(20 * 1024**3, 24 * 1024**3)):
            report = models.check_fit(spec, num_gpus=1)
            self.assertTrue(report.fits)
            self.assertEqual(report.placement, "single_gpu")
            self.assertEqual(report.chosen_gpu, 0)
            self.assertIsNone(report.max_memory)

    def test_check_fit_shards_across_two_gpus_when_neither_alone_suffices(self):
        spec = _fake_spec()
        # Each GPU has 8 GiB free; 10 GiB of weights needs both combined,
        # but no single GPU (8 GiB) could hold 10 GiB + headroom alone.
        with mock.patch.object(models, "estimate_weights_gib", return_value=self._fake_estimate(10.0)), \
             mock.patch("torch.cuda.mem_get_info", return_value=(8 * 1024 ** 3, 12 * 1024 ** 3)):
            report = models.check_fit(spec, num_gpus=2)
            self.assertTrue(report.fits)
            self.assertEqual(report.placement, "sharded")
            self.assertEqual(set(report.max_memory.keys()), {0, 1})
            self.assertIsNone(report.chosen_gpu)

    def test_check_fit_raises_when_even_sharded_across_two_gpus_does_not_fit(self):
        spec = _fake_spec(allow_cpu_offload=False)
        with mock.patch.object(models, "estimate_weights_gib", return_value=self._fake_estimate(100.0)), \
             mock.patch("torch.cuda.mem_get_info", return_value=(6 * 1024 ** 3, 8 * 1024 ** 3)):
            with self.assertRaises(ModelDoesNotFitError):
                models.check_fit(spec, num_gpus=2)

    def test_check_fit_falls_back_to_cpu_offload_when_allowed_and_sufficient(self):
        spec = _fake_spec(allow_cpu_offload=True)
        with mock.patch.object(models, "estimate_weights_gib", return_value=self._fake_estimate(20.0)), \
             mock.patch("torch.cuda.mem_get_info", return_value=(6 * 1024 ** 3, 8 * 1024 ** 3)), \
             mock.patch.object(environment, "cpu_free_ram_gib", return_value=50.0):
            report = models.check_fit(spec, num_gpus=2, allow_cpu_offload=True)
            self.assertTrue(report.fits)
            self.assertEqual(report.placement, "sharded_cpu_offload")
            self.assertIn("cpu", report.max_memory)
            self.assertIsNotNone(report.cpu_budget_gib)

    def test_check_fit_does_not_use_cpu_offload_unless_allowed(self):
        spec = _fake_spec(allow_cpu_offload=False)
        with mock.patch.object(models, "estimate_weights_gib", return_value=self._fake_estimate(20.0)), \
             mock.patch("torch.cuda.mem_get_info", return_value=(6 * 1024 ** 3, 8 * 1024 ** 3)), \
             mock.patch.object(environment, "cpu_free_ram_gib", return_value=50.0):
            with self.assertRaises(ModelDoesNotFitError):
                models.check_fit(spec, num_gpus=2, allow_cpu_offload=False)

    def test_check_disk_for_download_uses_monkeypatched_hf_api(self):
        spec = _fake_spec()
        fake_sibling = mock.Mock(rfilename="model.safetensors", size=2 * 1024**3)
        fake_info = mock.Mock(siblings=[fake_sibling])
        with mock.patch("huggingface_hub.HfApi.model_info", return_value=fake_info), \
             mock.patch.object(environment, "disk_free_gib", return_value=100.0):
            report = models.check_disk_for_download(spec)
            self.assertTrue(report["ok"])
        with mock.patch("huggingface_hub.HfApi.model_info", return_value=fake_info), \
             mock.patch.object(environment, "disk_free_gib", return_value=0.5):
            from optima_kaggle.errors import InsufficientDiskError
            with self.assertRaises(InsufficientDiskError):
                models.check_disk_for_download(spec)


class BuildBoundedMessagesTests(unittest.TestCase):
    def test_fits_without_reduction(self):
        fn = _minimal_base_json(1)["files"][0]["functions"][0]
        spec = _fake_spec(max_input_tokens=1000)
        messages, tokens, reductions = enrichment.build_bounded_messages(
            fn, FakeTokenizer(), spec, "colab_v2_no_module_ir"
        )
        self.assertEqual(reductions, [])
        self.assertLessEqual(tokens, spec.max_input_tokens)

    def test_ast_and_cfg_are_excluded_by_default_not_just_as_a_fallback(self):
        # AST/CFG make prompts large and are not part of the "useful local
        # information" a function-level prompt needs; they must be absent up
        # front, not only removed reactively once over budget.
        fn = _minimal_base_json(1)["files"][0]["functions"][0]
        fn["ast"] = "SomeAstNodeMarker " * 200
        fn["cfg"] = {"nodes": ["SomeCfgNodeMarker"] * 50}
        spec = _fake_spec(max_input_tokens=1000)
        messages, tokens, reductions = enrichment.build_bounded_messages(
            fn, FakeTokenizer(), spec, "colab_v2_no_module_ir"
        )
        combined = " ".join(m["content"] for m in messages)
        self.assertNotIn("SomeAstNodeMarker", combined)
        self.assertNotIn("SomeCfgNodeMarker", combined)
        self.assertEqual(reductions, [])  # excluded by default; no fallback reduction was needed

    def test_reduces_source_when_still_over_budget_after_defaults(self):
        fn = _minimal_base_json(1)["files"][0]["functions"][0]
        fn["source_code"] = "int x;\n" * 2000
        fn["calls"] = [f"call_target_{i}" for i in range(200)]
        spec = _fake_spec(max_input_tokens=700)
        messages, tokens, reductions = enrichment.build_bounded_messages(
            fn, FakeTokenizer(), spec, "colab_v2_no_module_ir"
        )
        self.assertLessEqual(tokens, spec.max_input_tokens)
        self.assertTrue(any(r.startswith("source_trimmed") for r in reductions))

    def test_prompt_too_long_raises_when_no_reduction_suffices(self):
        fn = _minimal_base_json(1)["files"][0]["functions"][0]
        fn["source_code"] = "verylongtoken " * 5000
        spec = _fake_spec(max_input_tokens=1)
        with self.assertRaises(PromptTooLongError):
            enrichment.build_bounded_messages(fn, FakeTokenizer(), spec, "colab_v2_no_module_ir")


class EnrichOneTests(unittest.TestCase):
    root = ARTIFACTS_ROOT / "enrich_one"

    def setUp(self):
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        self.fn = _minimal_base_json(1)["files"][0]["functions"][0]
        self.handle = _FakeHandle(_fake_spec(max_input_tokens=1000, max_new_tokens=32))
        self.gen = GenerationSettings(retries=2, max_new_tokens=32, prompt_variant="colab_v2_no_module_ir")
        self.attempts_log = self.root / "attempts.jsonl"

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_valid_json_on_first_attempt_completes(self):
        with mock.patch.object(models, "generate_text", return_value=_gen_result(VALID_ENRICHMENT_JSON)):
            record = enrichment.enrich_one(self.handle, self.fn, self.gen, self.attempts_log)
        self.assertEqual(record["status"], "completed")
        self.assertTrue(record["evaluation"]["request_success"])
        self.assertTrue(record["evaluation"]["json_valid"])
        self.assertEqual(record["evaluation"]["retry_count"], 0)
        self.assertIsNone(record["evaluation"]["failure_category"])
        self.assertTrue(self.attempts_log.exists())

    def test_fenced_json_is_parsed(self):
        fenced = f"```json\n{VALID_ENRICHMENT_JSON}\n```"
        with mock.patch.object(models, "generate_text", return_value=_gen_result(fenced)):
            record = enrichment.enrich_one(self.handle, self.fn, self.gen, self.attempts_log)
        self.assertEqual(record["status"], "completed")

    def test_garbage_output_exhausts_retries_and_preserves_raw_response(self):
        with mock.patch.object(models, "generate_text", return_value=_gen_result("not json at all")):
            record = enrichment.enrich_one(self.handle, self.fn, self.gen, self.attempts_log)
        self.assertEqual(record["status"], "failed")
        evaluation = record["evaluation"]
        self.assertEqual(evaluation["retry_count"], self.gen.retries)
        self.assertEqual(evaluation["failure_category"], "invalid_json")
        self.assertIn("not json at all", evaluation["raw_response"])
        # The failure reason must survive _evaluation()'s merge (regression
        # for the upstream bug where it was silently overwritten).
        self.assertIsNotNone(evaluation["failure_reason"])

    def test_schema_invalid_json_is_categorized_correctly(self):
        bad = json.dumps({"purpose": "x"})  # missing required fields
        with mock.patch.object(models, "generate_text", return_value=_gen_result(bad)):
            record = enrichment.enrich_one(self.handle, self.fn, self.gen, self.attempts_log)
        self.assertEqual(record["evaluation"]["failure_category"], "schema_invalid")

    def test_length_finish_with_bad_json_is_output_truncated(self):
        with mock.patch.object(models, "generate_text",
                               return_value=_gen_result("{incomplete", finish_reason="length")):
            record = enrichment.enrich_one(self.handle, self.fn, self.gen, self.attempts_log)
        self.assertEqual(record["evaluation"]["failure_category"], "output_truncated")

    def test_cuda_oom_gets_one_reduced_budget_retry_then_succeeds(self):
        import torch
        oom = torch.cuda.OutOfMemoryError("simulated oom")
        with mock.patch.object(models, "generate_text",
                               side_effect=[oom, _gen_result(VALID_ENRICHMENT_JSON)]):
            record = enrichment.enrich_one(self.handle, self.fn, self.gen, self.attempts_log)
        self.assertEqual(record["status"], "completed")
        lines = [json.loads(line) for line in self.attempts_log.read_text().splitlines()]
        self.assertEqual(lines[0]["category"], "cuda_oom")
        self.assertEqual(lines[1]["category"], "completed")

    def test_cuda_oom_persists_but_continues_when_gpu_recovers(self):
        import torch
        oom = torch.cuda.OutOfMemoryError("simulated oom")
        with mock.patch.object(models, "generate_text", side_effect=[oom, oom]), \
             mock.patch("torch.cuda.mem_get_info", return_value=(5 * 1024 ** 3, 8 * 1024 ** 3)):
            record = enrichment.enrich_one(self.handle, self.fn, self.gen, self.attempts_log)
        # A single OOM'd function must not raise/abort the run when the GPU
        # recovered real headroom afterward -- it is recorded as failed and
        # the caller (run_full_enrichment's loop) moves on to the next function.
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["evaluation"]["failure_category"], "cuda_oom")
        lines = [json.loads(line) for line in self.attempts_log.read_text().splitlines()]
        self.assertEqual([line["category"] for line in lines], ["cuda_oom", "cuda_oom"])

    def test_cuda_oom_aborts_when_gpu_still_unhealthy_after_retry(self):
        import torch
        oom = torch.cuda.OutOfMemoryError("simulated oom")
        with mock.patch.object(models, "generate_text", side_effect=[oom, oom]), \
             mock.patch("torch.cuda.mem_get_info", return_value=(0.1 * 1024 ** 3, 8 * 1024 ** 3)):
            with self.assertRaises(GenerationOOMError):
                enrichment.enrich_one(self.handle, self.fn, self.gen, self.attempts_log)

    def test_generation_error_is_retried_then_recovers(self):
        with mock.patch.object(models, "generate_text",
                               side_effect=[RuntimeError("transient"), _gen_result(VALID_ENRICHMENT_JSON)]):
            record = enrichment.enrich_one(self.handle, self.fn, self.gen, self.attempts_log)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["evaluation"]["retry_count"], 1)


class CheckpointTests(unittest.TestCase):
    root = ARTIFACTS_ROOT / "checkpoint"

    def setUp(self):
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        self.path = self.root / "test.checkpoint.jsonl"
        self.header = {"base_sha256": "abc123", "config_hash": "cfg1", "created_at": "now"}

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _completed_enrichment(self):
        return {**json.loads(VALID_ENRICHMENT_JSON), "model": "m", "status": "completed",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "evaluation": {"request_success": True, "json_valid": True, "retry_count": 0}}

    def test_append_and_is_completed(self):
        ckpt = Checkpoint(self.path)
        ckpt.open_or_create(self.header)
        self.assertFalse(ckpt.is_completed("f1"))
        ckpt.append("f1", self._completed_enrichment())
        self.assertTrue(ckpt.is_completed("f1"))

    def test_resume_loads_prior_records(self):
        ckpt = Checkpoint(self.path)
        ckpt.open_or_create(self.header)
        ckpt.append("f1", self._completed_enrichment())

        reloaded = Checkpoint(self.path)
        reloaded.open_or_create(self.header)
        self.assertTrue(reloaded.is_completed("f1"))
        self.assertEqual(reloaded.all_ids(), {"f1"})

    def test_header_mismatch_raises_resume_mismatch(self):
        ckpt = Checkpoint(self.path)
        ckpt.open_or_create(self.header)
        other = Checkpoint(self.path)
        with self.assertRaises(ResumeMismatchError):
            other.open_or_create({**self.header, "base_sha256": "different"})

    def test_torn_last_line_is_tolerated(self):
        ckpt = Checkpoint(self.path)
        ckpt.open_or_create(self.header)
        ckpt.append("f1", self._completed_enrichment())
        with self.path.open("a") as handle:
            handle.write('{"type": "function", "function_id": "f2", "incomple')  # no trailing newline
        reloaded = Checkpoint(self.path)
        reloaded.open_or_create(self.header)
        self.assertTrue(reloaded.is_completed("f1"))
        self.assertFalse(reloaded.is_completed("f2"))

    def test_malformed_middle_line_raises_corrupt(self):
        ckpt = Checkpoint(self.path)
        ckpt.open_or_create(self.header)
        ckpt.append("f1", self._completed_enrichment())
        with self.path.open("a") as handle:
            handle.write("not json at all\n")
            handle.write(json.dumps({"type": "function", "function_id": "f2",
                                     "enrichment": self._completed_enrichment(),
                                     "status": "completed"}) + "\n")
        reloaded = Checkpoint(self.path)
        with self.assertRaises(CheckpointCorruptError):
            reloaded.open_or_create(self.header)


class MaterializeTests(unittest.TestCase):
    root = ARTIFACTS_ROOT / "materialize"

    def setUp(self):
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        content = json.dumps(_minimal_base_json(function_count=2)).encode("utf-8")
        sha = hashlib.sha256(content).hexdigest()
        run_dir = self.root / "runs" / f"base-{sha[:12]}"
        base_path = run_dir / "base" / "base.json"
        base_path.parent.mkdir(parents=True)
        base_path.write_bytes(content)
        self.snap = snapshot.BaseSnapshot(path=base_path, sha256=sha, bytes=len(content),
                                          url="file:///fake", etag=None, downloaded_at="now",
                                          run_dir=run_dir)
        self.ctx = snapshot.RunContext.create(self.root, self.snap, {}, {}, "deadbeef")
        self.spec = _fake_spec(max_input_tokens=1000, slug="materialize-model")
        self.gen = GenerationSettings(retries=1, max_new_tokens=32)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_materialize_produces_valid_optima_artifact(self):
        from colab.colab_pipeline import flatten_functions, load_json, validate_json

        ckpt = Checkpoint(self.ctx.enriched_dir / f"{self.spec.slug}.checkpoint.jsonl")
        ckpt.open_or_create({"base_sha256": self.snap.sha256,
                             "config_hash": enrichment.config_hash(self.spec, self.gen),
                             "created_at": "now"})

        base = load_json(self.snap.path)
        functions = flatten_functions(base)
        handle = _FakeHandle(self.spec)
        attempts_log = self.ctx.logs_dir / self.spec.slug / "attempts.jsonl"
        with mock.patch.object(models, "generate_text", return_value=_gen_result(VALID_ENRICHMENT_JSON)):
            for fn in functions:
                record = enrichment.enrich_one(handle, fn, self.gen, attempts_log)
                ckpt.append(fn["id"], record)

        out_path = enrichment.materialize(self.ctx, self.snap, self.spec, self.gen, ckpt,
                                          elapsed=1.0, partial=False)
        info = validate_json(out_path)
        self.assertTrue(info["valid"])
        self.assertEqual(info["status"], "ENRICHED JSON")

        materialized = load_json(out_path)
        self.assertEqual(materialized["enrichment_metrics"]["functions_total"], 2)
        self.assertEqual(materialized["enrichment_metrics"]["functions_enriched"], 2)
        self.assertEqual(materialized["enrichment_metadata"]["model_slug"], self.spec.slug)


class RetrievalEvalTests(unittest.TestCase):
    root = ARTIFACTS_ROOT / "retrieval_eval"

    def setUp(self):
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        content = json.dumps(_minimal_base_json(function_count=2)).encode("utf-8")
        sha = hashlib.sha256(content).hexdigest()
        run_dir = self.root / "runs" / f"base-{sha[:12]}"
        base_path = run_dir / "base" / "base.json"
        base_path.parent.mkdir(parents=True)
        base_path.write_bytes(content)
        self.snap = snapshot.BaseSnapshot(path=base_path, sha256=sha, bytes=len(content),
                                          url="file:///fake", etag=None, downloaded_at="now",
                                          run_dir=run_dir)
        self.ctx = snapshot.RunContext.create(self.root, self.snap, {}, {}, "deadbeef")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_discover_passed_corpora_filters_and_prefixes_raw(self):
        self.ctx.set_model_status("m1", "enrichment_passed")
        self.ctx.set_model_status("m2", "enrichment_failed")
        corpora = retrieval_eval.discover_passed_corpora(self.ctx)
        self.assertEqual(corpora, ["raw", "m1"])

    def test_discover_passed_corpora_raises_when_none_passed(self):
        with self.assertRaises(NoCorporaError):
            retrieval_eval.discover_passed_corpora(self.ctx)

    def test_make_embedding_view_strips_failed_enrichments_only(self):
        from colab.colab_pipeline import flatten_functions, load_json, save_json

        base = load_json(self.snap.path)
        functions = flatten_functions(base)
        functions[0]["enrichment"] = {**json.loads(VALID_ENRICHMENT_JSON), "status": "completed"}
        functions[1]["enrichment"] = {"status": "failed", "purpose": "Insufficient implementation context."}
        save_json(base, self.ctx.enriched_dir / "enhanced_m1.json")

        view_path = retrieval_eval.make_embedding_view(self.ctx, "m1")
        view = load_json(view_path)
        view_functions = {f["id"]: f for f in flatten_functions(view)}
        self.assertIn("enrichment", view_functions[functions[0]["id"]])
        self.assertNotIn("enrichment", view_functions[functions[1]["id"]])

        manifest = json.loads((self.ctx.rag_inputs_dir / "m1.view.json").read_text())
        self.assertEqual(manifest["stripped_failed_ids"], [functions[1]["id"]])

    def test_build_indexes_refuses_mock_embeddings(self):
        with self.assertRaises(EmbeddingError):
            retrieval_eval.build_indexes(self.ctx, ["raw"], ["mock"], ["hybrid"])

    def test_build_comparison_rejects_mismatched_prompt_variants(self):
        for slug, variant in (("m1", "colab_v2"), ("m2", "colab_v2_no_module_ir")):
            (self.ctx.enriched_dir / f"enhanced_{slug}.json").write_text(json.dumps({
                "experiment": {"prompt_variant": variant, "do_sample": False, "max_new_tokens": 768,
                              "retries": 2, "base_sha256": self.snap.sha256, "model": slug},
                "enrichment_metadata": {"quantization": "nf4"},
                "enrichment_metrics": {"success_rate": 1.0, "functions_failed": 0},
            }))
        with self.assertRaises(ComparisonMismatchError):
            retrieval_eval.build_comparison(self.ctx, {}, ["raw", "m1", "m2"],
                                            {"manifest": {"sha256": "bench"}, "queries": []})


class GateBehaviorTests(unittest.TestCase):
    """Regression tests for two real failures observed on the live Kaggle
    2xT4 run: (1) a gate called with HANDLE=None raised a second, masking
    AttributeError instead of a clear prerequisite error; (2) GATE 4 failed
    outright when three functions completed successfully but the largest
    prompt's post-generation headroom was below the preferred 0.5 GiB.
    """

    root = ARTIFACTS_ROOT / "gate_behavior"

    def setUp(self):
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        content = json.dumps(_minimal_base_json(function_count=5)).encode("utf-8")
        sha = hashlib.sha256(content).hexdigest()
        run_dir = self.root / "runs" / f"base-{sha[:12]}"
        base_path = run_dir / "base" / "base.json"
        base_path.parent.mkdir(parents=True)
        base_path.write_bytes(content)
        self.snap = snapshot.BaseSnapshot(path=base_path, sha256=sha, bytes=len(content),
                                          url="file:///fake", etag=None, downloaded_at="now",
                                          run_dir=run_dir)
        self.ctx = snapshot.RunContext.create(self.root, self.snap, {}, {}, "deadbeef")
        self.spec = _fake_spec(max_input_tokens=1000, slug="gate-behavior-model")
        self.gen = GenerationSettings(retries=1, max_new_tokens=32)
        self.handle = _FakeHandle(self.spec)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    # --- (1) HANDLE=None must never produce a masking AttributeError ---

    def test_gate1_with_none_handle_raises_gate_failed_error(self):
        with self.assertRaises(GateFailedError):
            enrichment.gate1_load(self.ctx, None, self.gen)

    def test_gate2_with_none_handle_raises_gate_failed_error_not_attribute_error(self):
        with self.assertRaises(GateFailedError) as ctxmgr:
            enrichment.gate2_trivial(self.ctx, None, self.gen)
        self.assertIn("GATE 1", str(ctxmgr.exception))

    def test_gate3_with_none_handle_raises_gate_failed_error(self):
        with self.assertRaises(GateFailedError):
            enrichment.gate3_one_function(self.ctx, None, self.gen, self.snap)

    def test_gate4_with_none_handle_raises_gate_failed_error(self):
        with self.assertRaises(GateFailedError):
            enrichment.gate4_three_functions(self.ctx, None, self.gen, self.snap)

    def test_run_full_enrichment_with_none_handle_raises_gate_failed_error(self):
        with self.assertRaises(GateFailedError):
            enrichment.run_full_enrichment(self.ctx, None, self.gen, self.snap)

    # --- (2) GATE 4 no longer gates on GPU headroom at all ---

    @staticmethod
    def _distinct_enrichment_jsons():
        """Three schema-valid enrichments with distinct purposes, so
        purposes_not_all_same is satisfied the way three real, independently
        enriched functions would (rather than an identical mocked response).
        """
        base = json.loads(VALID_ENRICHMENT_JSON)
        return [json.dumps({**base, "purpose": f"Purpose {i}."}) for i in range(3)]

    def test_gate4_passes_regardless_of_low_headroom(self):
        # A low free_after_gib on a successfully completed function must never
        # fail (or warn on) GATE 4 -- three functions that actually completed
        # is what GATE 4 tests, not a theoretical GPU-headroom estimate.
        results = [_gen_result(text, free=0.05) for text in self._distinct_enrichment_jsons()]
        with mock.patch.object(models, "generate_text", side_effect=results):
            result = enrichment.gate4_three_functions(self.ctx, self.handle, self.gen, self.snap)
        self.assertTrue(result["passed"])
        self.assertEqual(result["details"]["status"], "PASSED")
        self.assertNotIn("resource_warnings", result["details"])
        self.assertNotIn("largest_prompt_headroom_ok", result["details"]["checks"])
        self.assertNotIn("largest_prompt_headroom_ok", result["details"])

    def test_gate4_passes_cleanly_when_headroom_is_fine(self):
        results = [_gen_result(text, free=5.0) for text in self._distinct_enrichment_jsons()]
        with mock.patch.object(models, "generate_text", side_effect=results):
            result = enrichment.gate4_three_functions(self.ctx, self.handle, self.gen, self.snap)
        self.assertTrue(result["passed"])
        self.assertEqual(result["details"]["status"], "PASSED")

    def test_gate4_still_fails_when_a_functional_check_actually_fails(self):
        with mock.patch.object(models, "generate_text", return_value=_gen_result("not json at all")):
            with self.assertRaises(GateFailedError):
                enrichment.gate4_three_functions(self.ctx, self.handle, self.gen, self.snap)

    # --- per-function GPU placement metadata ---

    def test_enrich_one_records_gpu_placement_per_function(self):
        handle = _FakeHandle(self.spec)
        handle.load_report = {"gpu_placement": "sharded", "gpu_count_used": 2, "used_cpu_offload": False}
        fn = _minimal_base_json(1)["files"][0]["functions"][0]
        with mock.patch.object(models, "generate_text", return_value=_gen_result(VALID_ENRICHMENT_JSON)):
            record = enrichment.enrich_one(handle, fn, self.gen, self.root / "attempts.jsonl")
        self.assertEqual(record["evaluation"]["gpu_placement"], "sharded")
        self.assertEqual(record["evaluation"]["gpu_count_used"], 2)
        self.assertFalse(record["evaluation"]["used_cpu_offload"])

    # --- GATE 4's conservative generation budget must not perturb config_hash ---

    def test_gate4_max_new_tokens_override_is_passed_to_generation_without_changing_config_hash(self):
        fn = _minimal_base_json(1)["files"][0]["functions"][0]
        captured_kwargs = {}

        def _capture(handle, messages, max_new_tokens, **kwargs):
            captured_kwargs["max_new_tokens"] = max_new_tokens
            return _gen_result(VALID_ENRICHMENT_JSON)

        expected_hash = enrichment.config_hash(self.spec, self.gen)
        with mock.patch.object(models, "generate_text", side_effect=_capture):
            record = enrichment.enrich_one(self.handle, fn, self.gen, self.root / "attempts.jsonl",
                                           max_new_tokens_override=256)
        self.assertEqual(captured_kwargs["max_new_tokens"], 256)
        self.assertEqual(record["evaluation"]["context_metrics"]["max_new_tokens"], 256)
        # gen.max_new_tokens itself, and therefore config_hash, is untouched.
        self.assertEqual(self.gen.max_new_tokens, 32)
        self.assertEqual(enrichment.config_hash(self.spec, self.gen), expected_hash)

    def test_gate4_three_functions_honors_max_new_tokens_override(self):
        captured = []
        texts = self._distinct_enrichment_jsons()

        def _capture(handle, messages, max_new_tokens, **kwargs):
            captured.append(max_new_tokens)
            return _gen_result(texts[len(captured) - 1])

        with mock.patch.object(models, "generate_text", side_effect=_capture):
            enrichment.gate4_three_functions(self.ctx, self.handle, self.gen, self.snap,
                                             max_new_tokens_override=256)
        self.assertEqual(captured, [256, 256, 256])


if __name__ == "__main__":
    unittest.main()
