#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from semantic_asr.reranking import (
    DEFAULT_FEATURE_SCHEMA,
    FeatureSchema,
    FeatureVector,
    RankingGroup,
    TrainingCandidate,
    train_constrained_linear_reranker,
)


def _finite(value: Any, *, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _read_jsonl(path: Path, schema: FeatureSchema) -> tuple[RankingGroup, ...]:
    groups: list[RankingGroup] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: group must be an object")
            group_id = str(row.get("groupId") or row.get("group_id") or "")
            candidate_rows = row.get("candidates")
            if not group_id or not isinstance(candidate_rows, list):
                raise ValueError(f"{path}:{line_number}: groupId and candidates are required")
            candidates: list[TrainingCandidate] = []
            for candidate_index, candidate_row in enumerate(candidate_rows):
                if not isinstance(candidate_row, dict):
                    raise ValueError(
                        f"{path}:{line_number}: candidate {candidate_index} must be an object"
                    )
                candidate_id = str(
                    candidate_row.get("candidateId") or candidate_row.get("candidate_id") or ""
                )
                raw_features = candidate_row.get("features")
                if not candidate_id or not isinstance(raw_features, dict):
                    raise ValueError(f"{path}:{line_number}: candidate ID/features are required")
                values = {
                    name: _finite(raw_features[name], name=f"feature:{name}")
                    for name in schema.names
                    if name in raw_features
                }
                features = FeatureVector.create(schema, values)
                candidates.append(
                    TrainingCandidate(
                        candidate_id=candidate_id,
                        features=features,
                        target_loss=_finite(
                            candidate_row.get("targetLoss", candidate_row.get("target_loss")),
                            name="targetLoss",
                        ),
                        critical_loss=_finite(
                            candidate_row.get(
                                "criticalLoss",
                                candidate_row.get("critical_loss", 0.0),
                            ),
                            name="criticalLoss",
                        ),
                        weight=_finite(candidate_row.get("weight", 1.0), name="weight"),
                    )
                )
            groups.append(
                RankingGroup(
                    group_id=group_id,
                    candidates=tuple(candidates),
                    metadata=dict(row.get("metadata") or {}),
                )
            )
    if not groups:
        raise ValueError(f"{path}: no ranking groups found")
    return tuple(groups)


def _load_schema(path: Path | None) -> FeatureSchema:
    if path is None:
        return DEFAULT_FEATURE_SCHEMA
    row = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(row, dict):
        raise ValueError("schema JSON must be an object")
    monotonicity = {
        str(name): int(value) for name, value in dict(row.get("monotonicity") or {}).items()
    }
    return FeatureSchema(
        names=tuple(str(value) for value in row["names"]),
        monotonicity=monotonicity,  # type: ignore[arg-type]
        version=str(row.get("version") or "1"),
    )


def _serialize_model(model: Any, report: Any) -> dict[str, object]:
    return {
        "format": "semantic-asr-constrained-linear-reranker-v1",
        "modelDigest": model.digest,
        "schema": {
            "names": list(model.schema.names),
            "monotonicity": model.schema.monotonicity,
            "version": model.schema.version,
            "digest": model.schema.digest,
        },
        "normalizer": {
            "means": model.normalizer.means,
            "scales": model.normalizer.scales,
            "schemaDigest": model.normalizer.schema_digest,
        },
        "weights": model.weights,
        "bias": model.bias,
        "objective": model.objective,
        "trainingDigest": model.training_digest,
        "metadata": model.metadata,
        "report": asdict(report),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the dependency-free constrained Semantic ASR v0.2 reranker "
            "from utterance-grouped JSONL."
        )
    )
    parser.add_argument("training_jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schema", type=Path)
    parser.add_argument(
        "--objective",
        choices=("pairwise", "listwise", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--critical-loss-weight", type=float, default=0.35)
    parser.add_argument("--listwise-weight", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    schema = _load_schema(args.schema)
    groups = _read_jsonl(args.training_jsonl, schema)
    model, report = train_constrained_linear_reranker(
        groups,
        schema=schema,
        objective=args.objective,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        critical_loss_weight=args.critical_loss_weight,
        listwise_weight=args.listwise_weight,
        seed=args.seed,
    )
    payload = _serialize_model(model, report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "modelDigest": model.digest,
                "groups": report.groups,
                "candidates": report.candidates,
                "pairwiseAccuracy": report.pairwise_accuracy,
                "pairwiseLoss": report.pairwise_loss,
                "listwiseLoss": report.listwise_loss,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
