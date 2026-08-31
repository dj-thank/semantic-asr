from __future__ import annotations

import json

import pytest

from semantic_asr.model_io import (
    deserialize_constrained_reranker,
    load_constrained_reranker,
    save_constrained_reranker,
    serialize_constrained_reranker,
)
from semantic_asr.reranking import (
    FeatureSchema,
    FeatureVector,
    RankingGroup,
    TrainingCandidate,
    train_constrained_linear_reranker,
)


def _model():
    schema = FeatureSchema(
        names=("acoustic", "missing"),
        monotonicity={"acoustic": 1, "missing": -1},
    )

    def vector(acoustic: float, missing: float) -> FeatureVector:
        return FeatureVector.create(
            schema,
            {"acoustic": acoustic, "missing": missing},
        )

    groups = (
        RankingGroup(
            "group-1",
            (
                TrainingCandidate("good", vector(2.0, 0.0), 0.0),
                TrainingCandidate("bad", vector(-1.0, 1.0), 1.0),
            ),
        ),
        RankingGroup(
            "group-2",
            (
                TrainingCandidate("good", vector(1.8, 0.0), 0.0),
                TrainingCandidate("bad", vector(-0.8, 0.9), 0.9),
            ),
        ),
    )
    model, report = train_constrained_linear_reranker(
        groups,
        schema=schema,
        epochs=80,
        seed=3,
    )
    return model, report, vector


def test_model_round_trip_and_file_loading(tmp_path) -> None:
    model, report, vector = _model()
    payload = serialize_constrained_reranker(
        model,
        report={"pairwiseAccuracy": report.pairwise_accuracy},
    )
    loaded = deserialize_constrained_reranker(payload)
    assert loaded.digest == model.digest
    assert loaded.rank({"good": vector(2.0, 0.0), "bad": vector(-1.0, 1.0)})[0][0] == "good"

    path = save_constrained_reranker(
        tmp_path / "reranker.json",
        model,
        report={"pairwiseAccuracy": report.pairwise_accuracy},
    )
    assert load_constrained_reranker(path).digest == model.digest


def test_model_digest_detects_tampering() -> None:
    model, _report, _vector = _model()
    payload = serialize_constrained_reranker(model)
    tampered = json.loads(json.dumps(payload))
    tampered["weights"]["acoustic"] += 0.5
    with pytest.raises(ValueError, match="digest mismatch"):
        deserialize_constrained_reranker(tampered)
