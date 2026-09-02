from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .benchmark import load_benchmark_jsonl, verify_split_isolation
from .offline_rerank import (
    build_calibration_samples,
    rerank_records,
    write_calibration_samples,
    write_reranked_benchmark,
)
from .ranker_calibration import RankerCalibrationProfile
from .rerankers import LinearCandidateRanker, LinearRankerProfile

EXPERIMENT_COMMANDS = {
    "apply-ranker",
    "partition-manifest",
    "score-ranker-calibration",
}


def _json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _linear_ranker(path: str | Path) -> LinearCandidateRanker:
    payload = _json(path)
    profile = LinearRankerProfile.from_dict(dict(payload.get("profile", payload)))
    return LinearCandidateRanker(profile)


def _calibration(path: str | Path) -> RankerCalibrationProfile:
    payload = _json(path)
    if payload.get("schemaVersion") != "ranker-calibration-v1":
        raise ValueError("ranker calibration artifact must use ranker-calibration-v1")
    return RankerCalibrationProfile.from_dict(dict(payload["profile"]))


def command_partition_manifest(args: argparse.Namespace) -> int:
    records = load_benchmark_jsonl(args.input)
    verify_split_isolation(records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "calibration", "test"):
        selected = [record for record in records if record.split == split]
        target = output_dir / f"{split}.jsonl"
        rows = []
        for record in selected:
            row = {
                "sampleId": record.sample_id,
                "groupId": record.group_id,
                "sourceId": record.source_id,
                "split": record.split,
                "reference": record.reference,
                "domain": record.domain,
                "nearDuplicateId": record.near_duplicate_id,
                "candidates": [candidate.as_dict() for candidate in record.candidates],
            }
            if record.annotated_reference is not None:
                row["annotatedReference"] = record.annotated_reference
            rows.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        counts[split] = len(selected)
    manifest = {
        "schemaVersion": "partition-manifest-v1",
        "source": str(Path(args.input)),
        "counts": counts,
        "groups": {
            split: len({record.group_id for record in records if record.split == split})
            for split in counts
        },
        "files": {split: str(output_dir / f"{split}.jsonl") for split in counts},
    }
    (output_dir / "partition.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def command_score_calibration(args: argparse.Namespace) -> int:
    records = load_benchmark_jsonl(args.input)
    ranker = _linear_ranker(args.ranker_profile)
    samples = build_calibration_samples(records, ranker)
    write_calibration_samples(samples, args.output)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(Path(args.output)),
                "samples": len(samples),
                "groups": len({sample.group_id for sample in samples}),
                "ranker": ranker.name,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_apply_ranker(args: argparse.Namespace) -> int:
    records = load_benchmark_jsonl(args.input)
    ranker = _linear_ranker(args.ranker_profile)
    calibration = _calibration(args.calibration) if args.calibration else None
    output = rerank_records(
        records,
        ranker,
        calibration=calibration,
        lexical_blend=args.lexical_blend,
    )
    write_reranked_benchmark(output, args.output)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(Path(args.output)),
                "samples": len(output),
                "ranker": ranker.name,
                "calibrationDigest": (calibration.digest if calibration is not None else None),
                "evidenceInjected": calibration is not None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic-asr",
        description="Leakage-safe Semantic ASR experiment manifest commands.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    partition = commands.add_parser("partition-manifest")
    partition.add_argument("input")
    partition.add_argument("--output-dir", required=True)
    partition.set_defaults(func=command_partition_manifest)

    calibration = commands.add_parser("score-ranker-calibration")
    calibration.add_argument("input", help="calibration split benchmark JSONL")
    calibration.add_argument("--ranker-profile", required=True)
    calibration.add_argument("--output", required=True)
    calibration.set_defaults(func=command_score_calibration)

    apply = commands.add_parser("apply-ranker")
    apply.add_argument("input")
    apply.add_argument("--ranker-profile", required=True)
    apply.add_argument("--calibration")
    apply.add_argument("--lexical-blend", type=float, default=0.65)
    apply.add_argument("--output", required=True)
    apply.set_defaults(func=command_apply_ranker)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
