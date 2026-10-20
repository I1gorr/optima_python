"""Command-line interface for the research pipelines (``python -m optima.rag <command>``).

Commands: ``build-benchmark``, ``validate-benchmark``, ``run-pipeline``, ``run-suite``,
``report``, ``rerankers``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from .benchmark import (BenchmarkError, build_developer_benchmark, build_structural_benchmark, load_benchmark,
                        validate_benchmark)
from .config import NAMED_PIPELINES, ConfigError, PipelineConfig, load_config_file
from .corpus import CorpusError, load_corpus

logger = logging.getLogger(__name__)


def _parse_overrides(items: List[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ConfigError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            overrides[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key.strip()] = raw
    return overrides


def cmd_build_benchmark(args: argparse.Namespace) -> int:
    corpus = load_corpus(Path(args.base_json))
    authored = json.loads(Path(args.authored).read_text(encoding="utf-8")) if args.authored else None
    if args.developer:
        if not authored:
            print("--developer needs --authored", file=sys.stderr)
            return 1
        bench = build_developer_benchmark(corpus, authored, benchmark_id=args.benchmark_id or "", version=args.version)
    else:
        bench = build_structural_benchmark(corpus, args.per_category, args.seed, authored,
                                           benchmark_id=args.benchmark_id or "", version=args.version)
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite {out}; pass --overwrite or choose a new --version/--out",
              file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bench, indent=1), encoding="utf-8")
    report = validate_benchmark(load_benchmark(out), corpus)
    print(f"wrote {out}: {len(bench['queries'])} queries {bench['counts']}")
    for category, reason in bench["unsupported_categories"].items():
        print(f"  unsupported category '{category}': {reason}")
    print(f"validation ok={report['ok']}")
    return 0 if report["ok"] else 1


def cmd_validate_benchmark(args: argparse.Namespace) -> int:
    corpus = load_corpus(Path(args.base_json))
    report = validate_benchmark(load_benchmark(Path(args.benchmark)), corpus)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def cmd_run_pipeline(args: argparse.Namespace) -> int:
    from .dense import EmbeddingCache
    from .experiment import ExperimentStore, run_pipeline
    overrides = _parse_overrides(args.set)
    if not args.config and not args.base_json:
        raise ConfigError("--base-json is required (or pass a --config containing base_json)")
    if args.embedding_model:
        overrides["embedding_model"] = args.embedding_model
    if args.config:
        config = load_config_file(Path(args.config)).replaced(overrides)
    else:
        factory = NAMED_PIPELINES[args.pipeline]
        if args.pipeline == "baseline_rag":
            config = factory(args.base_json, **overrides)
        else:
            if not args.enriched_json:
                raise ConfigError(f"{args.pipeline} requires --enriched-json")
            config = factory(args.base_json, args.enriched_json, args.enrichment_model or "",
                             **overrides)
    bench = load_benchmark(Path(args.benchmark))
    cache = EmbeddingCache(Path(args.results_dir) / "_embedding_cache")
    result = run_pipeline(config, bench, embedding_cache=cache, suite_id=args.suite_id,
                          label=args.label or config.name, group="adhoc",
                          bootstrap_samples=args.bootstrap_samples)
    store = ExperimentStore(Path(args.results_dir), args.suite_id)
    run_dir = store.save_run(result)
    overall = result["metrics"]["overall"]
    print(f"experiment {result['experiment_id']} [{result['status']}] saved to {run_dir}")
    print("  " + "  ".join(f"{k}={overall[k]:.3f}" for k in
                           ("mrr", "ndcg_at_10", "recall_at_1", "recall_at_5", "recall_at_10")))
    for category, values in result["metrics"]["by_category"].items():
        print(f"  {category:<16} n={values['n_queries']:<3} mrr={values['mrr']:.3f} "
              f"r@10={values['recall_at_10']:.3f}")
    return 0 if result["status"] == "ok" else 2


def cmd_run_suite(args: argparse.Namespace) -> int:
    from .suite import load_suite_spec, run_suite
    spec = load_suite_spec(Path(args.spec))
    outcome = run_suite(spec, Path(args.results_dir) if args.results_dir else None,
                        resume=args.resume, only_groups=args.groups, dry_run=args.dry_run)
    print(json.dumps(outcome, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .experiment import ExperimentStore
    from .reporting import write_suite_reports
    suite_dir = Path(args.suite_dir)
    store = ExperimentStore(suite_dir.parent, suite_dir.name)
    info_path = suite_dir / "suite.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    paths = write_suite_reports(store, info, [], args.bootstrap_samples)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return 0


def cmd_rerankers(args: argparse.Namespace) -> int:
    from .rerank import RERANKER_REGISTRY
    for spec in RERANKER_REGISTRY.values():
        print(f"{spec.alias:<20} {spec.model_name:<42} {spec.license:<11} {spec.parameters_m:>6.1f}M  {spec.hardware}")
    return 0


def register(subparsers: Any) -> None:
    """Attach the research commands to an existing ``argparse`` sub-parser group."""
    p = subparsers.add_parser("build-benchmark", help="Generate a labelled structural benchmark from base.json")
    p.add_argument("--base-json", required=True)
    p.add_argument("--authored", help="hand-authored behavioural queries JSON")
    p.add_argument("--developer", action="store_true",
                   help="build a developer-intent benchmark from --authored only (no generated structural queries)")
    p.add_argument("--out", required=True)
    p.add_argument("--per-category", type=int, default=20)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--benchmark-id")
    p.add_argument("--version", default="1.0.0")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=guarded(cmd_build_benchmark))

    p = subparsers.add_parser("validate-benchmark", help="Check benchmark IDs against base.json")
    p.add_argument("--base-json", required=True)
    p.add_argument("--benchmark", required=True)
    p.set_defaults(func=guarded(cmd_validate_benchmark))

    p = subparsers.add_parser(
        "run-pipeline", help="Run one pipeline (baseline_rag | semantic_enrichment_rag | "
                             "compiler_hybrid_rag, or --config FILE) on a benchmark")
    p.add_argument("--pipeline", choices=sorted(NAMED_PIPELINES), default="baseline_rag")
    p.add_argument("--config", help="pipeline config JSON (overrides --pipeline)")
    p.add_argument("--base-json", help="base.json (raw/compiler data and the ID universe)")
    p.add_argument("--enriched-json")
    p.add_argument("--enrichment-model", help="label recorded in results")
    p.add_argument("--embedding-model", help="registry alias or HF id (e.g. bge-small)")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, e.g. --set retrieval.bm25=true --set graph.max_depth=2")
    p.add_argument("--results-dir", default="experiments")
    p.add_argument("--suite-id", default="adhoc")
    p.add_argument("--label")
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.set_defaults(func=guarded(cmd_run_pipeline))

    p = subparsers.add_parser("run-suite", help="Run the full ablation benchmark from a suite spec")
    p.add_argument("--spec", required=True)
    p.add_argument("--results-dir")
    p.add_argument("--groups", nargs="+", help="restrict to these ablation groups")
    p.add_argument("--resume", action="store_true", help="skip runs already completed in this suite")
    p.add_argument("--dry-run", action="store_true", help="list planned runs and exit")
    p.set_defaults(func=guarded(cmd_run_suite))

    p = subparsers.add_parser("report", help="Regenerate tables, comparisons, plots and report.md")
    p.add_argument("--suite-dir", required=True)
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.set_defaults(func=guarded(cmd_report))

    p = subparsers.add_parser("rerankers", help="List supported cross-encoders, licences, hardware")
    p.set_defaults(func=guarded(cmd_rerankers))


COMMANDS = {"build-benchmark", "validate-benchmark", "run-pipeline", "run-suite", "report",
            "rerankers"}


def guarded(func):
    """Wrap a command so expected user errors print a clear message instead of a traceback."""
    def wrapper(args: argparse.Namespace) -> None:
        try:
            code = func(args)
        except (ConfigError, CorpusError, BenchmarkError, FileExistsError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            from .dense import EmbeddingUnavailable
            from .rerank import RerankerUnavailable
            if isinstance(exc, (EmbeddingUnavailable, RerankerUnavailable)):
                print(f"error: {exc}", file=sys.stderr)
                sys.exit(3)
            raise
        sys.exit(code or 0)
    return wrapper
