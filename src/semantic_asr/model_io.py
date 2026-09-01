from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .reranking import (
    ConstrainedLinearReranker,
    FeatureNormalizer,
    FeatureSchema,
)

MODEL_FORMAT = "semantic-asr-constrained-linear-reranker-v1"


def serialize_constrained_reranker(
    model: ConstrainedLinearReranker,
    *,
    report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "format": MODEL_FORMAT,
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
        "report": dict(report or {}),
    }


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def deserialize_constrained_reranker(
    payload: Mapping[str, object],
    *,
    verify_digest: bool = True,
) -> ConstrainedLinearReranker:
    if payload.get("format") != MODEL_FORMAT:
        raise ValueError("unsupported constrained reranker format")
    schema_row = _mapping(payload.get("schema"), name="schema")
    monotonicity = {
        str(name): int(value)
        for name, value in _mapping(
            schema_row.get("monotonicity", {}),
            name="schema.monotonicity",
        ).items()
    }
    schema = FeatureSchema(
        names=tuple(str(value) for value in schema_row["names"]),
        monotonicity=monotonicity,  # type: ignore[arg-type]
        version=str(schema_row.get("version") or "1"),
    )
    declared_schema_digest = schema_row.get("digest")
    if declared_schema_digest is not None and str(declared_schema_digest) != schema.digest:
        raise ValueError("feature schema digest mismatch")

    normalizer_row = _mapping(payload.get("normalizer"), name="normalizer")
    means = {
        str(name): float(value)
        for name, value in _mapping(normalizer_row.get("means"), name="normalizer.means").items()
    }
    scales = {
        str(name): float(value)
        for name, value in _mapping(normalizer_row.get("scales"), name="normalizer.scales").items()
    }
    if set(means) != set(schema.names) or set(scales) != set(schema.names):
        raise ValueError("normalizer does not match feature schema")
    if any(not math.isfinite(value) for value in (*means.values(), *scales.values())):
        raise ValueError("normalizer values must be finite")
    if any(value <= 0 for value in scales.values()):
        raise ValueError("normalizer scales must be positive")
    normalizer = FeatureNormalizer(
        means=means,
        scales=scales,
        schema_digest=str(normalizer_row.get("schemaDigest") or ""),
    )
    if normalizer.schema_digest != schema.digest:
        raise ValueError("normalizer schema digest mismatch")

    weights = {
        str(name): float(value)
        for name, value in _mapping(payload.get("weights"), name="weights").items()
    }
    objective = str(payload.get("objective") or "")
    if objective not in {"pairwise", "listwise", "hybrid"}:
        raise ValueError("unsupported constrained reranker objective")
    model = ConstrainedLinearReranker(
        schema=schema,
        normalizer=normalizer,
        weights=weights,
        bias=float(payload.get("bias")),
        objective=objective,  # type: ignore[arg-type]
        training_digest=str(payload.get("trainingDigest") or ""),
        metadata=dict(_mapping(payload.get("metadata", {}), name="metadata")),
    )
    declared_digest = payload.get("modelDigest")
    if verify_digest and str(declared_digest or "") != model.digest:
        raise ValueError("constrained reranker model digest mismatch")
    return model


def save_constrained_reranker(
    path: str | Path,
    model: ConstrainedLinearReranker,
    *,
    report: Mapping[str, object] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            serialize_constrained_reranker(model, report=report),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def load_constrained_reranker(
    path: str | Path,
    *,
    verify_digest: bool = True,
) -> ConstrainedLinearReranker:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("model file must contain a JSON object")
    return deserialize_constrained_reranker(payload, verify_digest=verify_digest)
