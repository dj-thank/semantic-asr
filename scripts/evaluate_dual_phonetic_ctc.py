#!/usr/bin/env python3
"""Evaluate a frozen dual phone/mora CTC artifact on one held-out manifest split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_asr.phonetic_runtime.evaluation import evaluate_phonetic_runtime
from semantic_asr.phonetic_runtime.inference import DualCTCPosteriorRuntime
from semantic_asr.phonetic_runtime.manifest import load_phonetic_manifest


def _outside_checkout(path: Path) -> Path:
    destination = path.resolve()
    repository = Path(__file__).resolve().parents[1]
    if destination == repository or repository in destination.parents:
        raise ValueError("evaluation reports must be written outside the checkout")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "calibration", "test"), default="test"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-predictions",
        action="store_true",
        help="Write raw phone/mora predictions. The default report contains hashes only.",
    )
    args = parser.parse_args()

    output = _outside_checkout(args.output)
    runtime = DualCTCPosteriorRuntime.from_artifact(args.artifact_dir, device=args.device)
    manifest = load_phonetic_manifest(args.manifest)
    report = evaluate_phonetic_runtime(runtime, manifest, split=args.split)
    report.write(output, include_predictions=args.include_predictions)
    print(
        json.dumps(
            {
                "report": str(output),
                "reportDigest": report.digest,
                "runtimeProfileDigest": runtime.profile_digest,
                "manifestDigest": manifest.digest,
                "split": args.split,
                "phoneErrorRate": report.phone_error_rate,
                "moraErrorRate": report.mora_error_rate,
                "utterances": len(report.utterances),
                "totalLatencyMs": report.total_latency_ms,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
