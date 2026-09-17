"""Kaggle/Python environment diagnostics, dependency checks and bitsandbytes
compatibility probing.

Module-level imports are stdlib-only on purpose: this module must be
importable (and ``diagnose``-able for CUDA presence) before any heavy
dependency such as torch has been confirmed to work. Every function that
needs torch/transformers imports it locally.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from .errors import EnvironmentError_, QuantizationUnavailableError

# import name -> pip requirement string (kept in sync with pyproject.toml)
_REQUIRED_PACKAGES: dict[str, str] = {
    "langchain_core": "langchain-core>=0.1.0",
    "langchain_community": "langchain-community>=0.0.10",
    "langchain_text_splitters": "langchain-text-splitters>=0.0.1",
    "langchain_huggingface": "langchain-huggingface>=0.0.1",
    "faiss": "faiss-cpu>=1.7.4",
    "numpy": "numpy>=1.24.0",
    "accelerate": "accelerate>=0.30.0",
    "sentence_transformers": "sentence-transformers>=2.7.0",
    "tqdm": "tqdm>=4.66.0",
    "transformers": "transformers>=4.40.0",
    "matplotlib": "matplotlib>=3.7.0",
    "packaging": "packaging",
    "huggingface_hub": "huggingface_hub",
}

_MIN_TRANSFORMERS = (4, 45)


def _pkg_version(import_name: str) -> Optional[str]:
    try:
        module = importlib.import_module(import_name)
    except ImportError:
        return None
    return getattr(module, "__version__", None) or "unknown"


def _parse_version(text: str) -> tuple[int, ...]:
    parts = []
    for chunk in text.split(".")[:3]:
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def ensure_python_deps(allow_install: bool = True) -> dict[str, Any]:
    """Check every required import; install only what is missing.

    Never touches torch. Never installs clang. Never upgrades/downgrades an
    already-installed package (transformers is checked for a minimum version
    but is never auto-upgraded).
    """
    found: dict[str, Optional[str]] = {name: _pkg_version(name) for name in _REQUIRED_PACKAGES}
    missing = [name for name, version in found.items() if version is None]

    installed_now: list[str] = []
    if missing:
        if not allow_install:
            raise EnvironmentError_(
                f"Missing required packages {missing} and allow_install=False. "
                f"Install them manually, e.g.: pip install "
                f"{' '.join(_REQUIRED_PACKAGES[name] for name in missing)}"
            )
        requirements = [_REQUIRED_PACKAGES[name] for name in missing]
        print(f"Installing missing packages: {requirements}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *requirements],
            capture_output=True, text=True, timeout=900,
        )
        if result.returncode != 0:
            raise EnvironmentError_(
                "pip install failed for missing dependencies "
                f"{requirements}:\n{result.stderr[-3000:]}"
            )
        for name in missing:
            found[name] = _pkg_version(name)
            installed_now.append(name)
        still_missing = [name for name in missing if found[name] is None]
        if still_missing:
            raise EnvironmentError_(
                f"pip install reported success but these imports still fail: "
                f"{still_missing}. Check for a naming mismatch or a broken wheel."
            )

    transformers_version = found.get("transformers")
    if transformers_version and transformers_version != "unknown":
        if _parse_version(transformers_version) < _MIN_TRANSFORMERS:
            raise EnvironmentError_(
                f"transformers=={transformers_version} is older than the required "
                f"4.45+ (needed so generate() does not materialize full-sequence "
                f"logits during prefill, which can OOM on its own). Restart the "
                f"session and run: pip install -q -U 'transformers>=4.45'. "
                f"This tool will not upgrade it automatically."
            )

    report = {"found": found, "installed_now": installed_now}
    for name, version in found.items():
        marker = " (installed now)" if name in installed_now else ""
        print(f"  {name:<24} {version}{marker}")
    return report


def _cpu_ram_gib() -> Optional[float]:
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 * 1024), 2)
    except OSError:
        pass
    return None


def cpu_free_ram_gib() -> Optional[float]:
    """Currently available (not just free) CPU RAM, for CPU-offload budgeting.

    ``MemAvailable`` (not ``MemFree``) is used because it already accounts
    for reclaimable cache/buffers, which is what actually matters for "how
    much RAM could a new large allocation use."
    """
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 * 1024), 2)
    except OSError:
        pass
    return None


def _internet_reachable(url: str = "https://huggingface.co", timeout: float = 5.0) -> bool:
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def diagnose(hf_home: Optional[str] = None, working_dir: str = "/kaggle/working") -> dict[str, Any]:
    """Report the environment and hard-stop if CUDA is unavailable."""
    import platform

    import torch

    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cpu_ram_gib": _cpu_ram_gib(),
        "internet_reachable": _internet_reachable(),
    }
    if not info["cuda_available"]:
        raise EnvironmentError_(
            "CUDA is unavailable. Enable a GPU runtime (Settings > Accelerator > "
            "GPU T4 x2 or P100) before running the model-loading cells."
        )
    if not info["internet_reachable"]:
        raise EnvironmentError_(
            "huggingface.co is not reachable. Enable Internet in the notebook's "
            "Settings panel before running the model-loading cells."
        )

    num_gpus = torch.cuda.device_count()
    gpus = []
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        gpus.append({
            "index": i, "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_vram_gib": round(total_bytes / (1024 ** 3), 3),
            "free_vram_gib": round(free_bytes / (1024 ** 3), 3),
        })
    info.update({
        "num_gpus": num_gpus, "multi_gpu_capable": num_gpus > 1, "gpus": gpus,
        # Back-compat single-GPU convenience fields, mirroring GPU 0.
        "gpu_name": gpus[0]["name"] if gpus else None,
        "compute_capability": gpus[0]["compute_capability"] if gpus else None,
        "gpu_total_vram_gib": gpus[0]["total_vram_gib"] if gpus else None,
        "gpu_free_vram_gib": gpus[0]["free_vram_gib"] if gpus else None,
    })

    for label, path in (("working_dir", working_dir), ("hf_home", hf_home or os.environ.get("HF_HOME", ""))):
        if path:
            try:
                usage = shutil.disk_usage(path if Path(path).exists() else Path(path).parent)
                info[f"{label}_disk_free_gib"] = round(usage.free / (1024 ** 3), 2)
            except OSError:
                info[f"{label}_disk_free_gib"] = None

    for key, value in info.items():
        if key == "gpus":
            for gpu in value:
                print(f"  gpu[{gpu['index']}]                   {gpu['name']}, "
                      f"cc={gpu['compute_capability']}, "
                      f"free={gpu['free_vram_gib']}/{gpu['total_vram_gib']} GiB")
            continue
        print(f"  {key:<26} {value}")
    return info


@dataclass
class BnbStatus:
    ok: bool
    version: Optional[str]
    stage: str
    message: str
    stdout_tail: str = ""
    stderr_tail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_BNB_PROBE_SCRIPT = """
import json
import sys
try:
    import torch
    import bitsandbytes as bnb
    layer = bnb.nn.Linear4bit(
        256, 256, bias=False, compute_dtype=torch.float16, quant_type="nf4"
    ).cuda()
    x = torch.randn(4, 256, dtype=torch.float16, device="cuda")
    y = layer(x)
    assert torch.isfinite(y).all(), "non-finite output from Linear4bit"
    print(json.dumps({
        "ok": True,
        "torch_version": torch.__version__,
        "bnb_version": getattr(bnb, "__version__", "unknown"),
        "cuda_version": torch.version.cuda,
    }))
except Exception as exc:  # noqa: BLE001 - this is a diagnostic subprocess
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
    sys.exit(1)
"""


def probe_bitsandbytes(allow_install: bool = True, min_version: str = "0.43.0") -> BnbStatus:
    """Check that bitsandbytes is installed, importable, and functional on CUDA.

    Never silently falls back to fp16 and never downgrades an existing
    installation. Runs the functional check in a subprocess so a broken
    native library cannot crash the notebook kernel.
    """
    try:
        version = importlib.metadata.version("bitsandbytes")
    except importlib.metadata.PackageNotFoundError:
        version = None

    if version is None:
        if not allow_install:
            return BnbStatus(False, None, "not_installed",
                              "bitsandbytes is not installed and allow_install=False.")
        pre = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__)"],
                              capture_output=True, text=True, timeout=60)
        torch_before = pre.stdout.strip()
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.45.0"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            return BnbStatus(False, None, "install_failed",
                              "pip install bitsandbytes failed.",
                              result.stdout[-2000:], result.stderr[-2000:])
        post = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__)"],
                               capture_output=True, text=True, timeout=60)
        torch_after = post.stdout.strip()
        if torch_before and torch_after and torch_before != torch_after:
            raise QuantizationUnavailableError(
                f"Installing bitsandbytes changed the torch version "
                f"({torch_before} -> {torch_after}). This can silently break the "
                f"preinstalled CUDA build. Restart the session and do not enable "
                f"nf4 quantization, or pin a bitsandbytes version compatible with "
                f"the existing torch build."
            )
        try:
            version = importlib.metadata.version("bitsandbytes")
        except importlib.metadata.PackageNotFoundError:
            return BnbStatus(False, None, "install_failed",
                              "bitsandbytes still not importable after install.")

    if _parse_version(version) < _parse_version(min_version):
        return BnbStatus(
            False, version, "too_old",
            f"bitsandbytes=={version} is older than the required {min_version}+. "
            f"This tool will not upgrade it automatically. If you want nf4, run: "
            f"pip install -q -U 'bitsandbytes>=0.45.0' and restart the session.",
        )

    probe = subprocess.run(
        [sys.executable, "-c", _BNB_PROBE_SCRIPT],
        capture_output=True, text=True, timeout=180,
    )
    stdout_tail = probe.stdout[-2000:]
    stderr_tail = probe.stderr[-2000:]
    if probe.returncode != 0:
        payload: dict[str, Any] = {}
        try:
            payload = json.loads(probe.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            pass
        error = payload.get("error", "no diagnostic output; see stderr_tail")
        stage = "cuda_kernel_failed" if "CUDA" in error or "cuda" in error else "import_failed"
        return BnbStatus(False, version, stage,
                          f"bitsandbytes functional probe failed: {error}",
                          stdout_tail, stderr_tail)

    try:
        payload = json.loads(probe.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return BnbStatus(False, version, "import_failed",
                          "bitsandbytes probe produced no parsable output.",
                          stdout_tail, stderr_tail)

    try:
        from transformers.utils import is_bitsandbytes_available
        if not is_bitsandbytes_available():
            return BnbStatus(False, version, "transformers_incompatible",
                              "transformers.utils.is_bitsandbytes_available() is "
                              "False even though the subprocess probe passed.",
                              stdout_tail, stderr_tail)
    except ImportError:
        pass

    return BnbStatus(True, version, "ok", "bitsandbytes nf4 is usable on this GPU.",
                      stdout_tail, stderr_tail)


def require_bitsandbytes(status: BnbStatus, spec_slug: str) -> None:
    if status.ok:
        return
    raise QuantizationUnavailableError(
        f"Model {spec_slug!r} requires nf4 quantization but bitsandbytes is not "
        f"usable (stage={status.stage}): {status.message}\n"
        f"stderr tail: {status.stderr_tail}\n"
        f"Actions: (a) choose an fp16-feasible model such as "
        f"'qwen25-3b-instruct-fp16', or (b) restart the session and install a "
        f"bitsandbytes build compatible with the preinstalled torch/CUDA."
    )


def gpu_report(label: str, log_path: Optional[Path] = None) -> dict[str, Any]:
    """Print current GPU memory stats for every visible GPU (plus an
    aggregate) and optionally append the record to a jsonl log.

    Never reads ``torch.cuda.memory_allocated()``/``mem_get_info()`` with no
    argument: that silently means "current device only," which is misleading
    once a model is sharded across more than one GPU.
    """
    import torch

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus == 0:
        record = {"label": label, "cuda_available": False, "timestamp": time.time()}
        print(f"[gpu:{label}] cuda_available=False")
        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        return record

    gpus = []
    for i in range(num_gpus):
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        gpus.append({
            "index": i,
            "allocated_gib": round(torch.cuda.memory_allocated(i) / (1024 ** 3), 4),
            "reserved_gib": round(torch.cuda.memory_reserved(i) / (1024 ** 3), 4),
            "max_allocated_gib": round(torch.cuda.max_memory_allocated(i) / (1024 ** 3), 4),
            "free_gib": round(free_bytes / (1024 ** 3), 4),
            "total_gib": round(total_bytes / (1024 ** 3), 4),
        })
    aggregate = {
        "allocated_gib": round(sum(g["allocated_gib"] for g in gpus), 4),
        "reserved_gib": round(sum(g["reserved_gib"] for g in gpus), 4),
        "max_allocated_gib": round(sum(g["max_allocated_gib"] for g in gpus), 4),
        "free_gib": round(min(g["free_gib"] for g in gpus), 4),
        "total_gib": round(sum(g["total_gib"] for g in gpus), 4),
    }
    record = {"label": label, "gpus": gpus, "aggregate": aggregate, "timestamp": time.time()}

    per_gpu_text = " | ".join(
        f"gpu{g['index']}: alloc={g['allocated_gib']} free={g['free_gib']}/{g['total_gib']}"
        for g in gpus
    )
    print(f"[gpu:{label}] {per_gpu_text} || total_alloc={aggregate['allocated_gib']}, "
          f"min_free={aggregate['free_gib']}")
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    return record


def assert_gpu_clean(threshold_gib: float = 0.3) -> None:
    """Check every visible GPU, not just the current device. A model sharded
    across two GPUs that only unloads GPU 0 must still be caught here.
    """
    import gc

    import torch

    from .errors import GpuNotCleanError

    dirty = []
    for i in range(torch.cuda.device_count()):
        allocated_gib = torch.cuda.memory_allocated(i) / (1024 ** 3)
        if allocated_gib >= threshold_gib:
            dirty.append((i, round(allocated_gib, 3)))
    if not dirty:
        return
    culprits = []
    try:
        for obj in gc.get_objects():
            type_name = type(obj).__name__
            if type_name in ("PreTrainedModel",) or "Model" in type_name and hasattr(obj, "parameters"):
                culprits.append(type_name)
    except Exception:  # noqa: BLE001 - best-effort diagnostic only
        pass
    dirty_text = ", ".join(f"GPU {i}: {gib} GiB" for i, gib in dirty)
    raise GpuNotCleanError(
        f"GPU memory was not released on {len(dirty)} device(s) (threshold "
        f"{threshold_gib} GiB): {dirty_text}. Live model-like objects found: "
        f"{sorted(set(culprits)) or 'none found by gc scan'}. "
        f"A stale reference (e.g. a notebook Out[] cache or sys.last_traceback) "
        f"is likely keeping the model alive."
    )


def disk_free_gib(path: str | Path) -> float:
    path = Path(path)
    usage = shutil.disk_usage(path if path.exists() else path.parent)
    return round(usage.free / (1024 ** 3), 3)
