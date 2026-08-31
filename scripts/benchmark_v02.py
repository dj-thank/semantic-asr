#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from semantic_asr.experiment import SampleResult, paired_bootstrap_comparison


def _read_results(path: Path) -> tuple[SampleResult, ...]:
    rows: list[SampleResult] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: result must be an object")
            metrics = {
                str(name): float(value) for name, value in dict(row.get("metrics") or {}).items()
            }
            if any(not math.isfinite(value) for value in metrics.values()):
                raise ValueError(f"{path}:{line_number}: metrics must be finite")
            rows.append(
                SampleResult(
                    sample_id=str(row.get("sampleId") or row.get("sample_id") or ""),
                    system_id=str(row.get("systemId") or row.get("system_id") or ""),
                    metrics=metrics,
                    latency_ms=float(row.get("latencyMs", row.get("latency_ms", 0.0))),
                    peak_memory_mb=(
                        None
                        if row.get("peakMemoryMb", row.get("peak_memory_mb")) is None
                        else float(row.get("peakMemoryMb", row.get("peak_memory_mb")))
                    ),
                    accepted=bool(row.get("accepted", True)),
                    metadata=dict(row.get("metadata") or {}),
                )
            )
    if not rows:
        raise ValueError(f"{path}: no benchmark results found")
    return tuple(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a paired bootstrap comparison for Semantic ASR v0.2 JSONL results."
    )
    parser.add_argument("results_jsonl", type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Treat larger metric values as better. Loss/error metrics default to lower-is-better.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    comparison = paired_bootstrap_comparison(
        _read_results(args.results_jsonl),
        baseline_system=args.baseline,
        candidate_system=args.candidate,
        metric=args.metric,
        iterations=args.iterations,
        confidence=args.confidence,
        seed=args.seed,
        lower_is_better=not args.higher_is_better,
    )
    payload = asdict(comparison)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
