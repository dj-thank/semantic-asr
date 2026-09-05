from __future__ import annotations

import json

import pytest

from semantic_asr.document_experiment.artifacts import BidirectionalNgramArtifact
from semantic_asr.document_experiment.ngram_scorer import (
    NgramCalibrationSequence,
    fit_character_ngram_model,
    fit_ngram_normalization,
)

TRAIN = "1" * 64
CALIBRATION = "2" * 64


def artifact() -> BidirectionalNgramArtifact:
    texts = (
        "レビュー完了まではまだマージしません。",
        "承認後に統合します。",
        "確認中なのでまだ変更しません。",
    )
    forward = fit_character_ngram_model(
        texts,
        order=3,
        alpha=0.2,
        training_manifest_sha256=TRAIN,
        revision="forward-r1",
    )
    backward = fit_character_ngram_model(
        texts,
        order=3,
        alpha=0.2,
        training_manifest_sha256=TRAIN,
        revision="backward-r1",
        reversed_text=True,
    )
    normalization = fit_ngram_normalization(
        forward,
        backward,
        (
            NgramCalibrationSequence(text=texts[0]),
            NgramCalibrationSequence(text=texts[1]),
        ),
        calibration_manifest_sha256=CALIBRATION,
        revision="cal-r1",
    )
    return BidirectionalNgramArtifact(
        forward=forward,
        backward=backward,
        normalization=normalization,
        name="fixture",
        revision="artifact-r1",
    )


def test_artifact_round_trip_and_scorer_identity(tmp_path) -> None:
    original = artifact()
    destination = original.write(tmp_path / "ngram.json")

    loaded = BidirectionalNgramArtifact.read(destination)

    assert loaded == original
    assert loaded.digest == original.digest
    assert loaded.scorer().profile_digest == original.scorer().profile_digest


def test_artifact_tampering_is_detected(tmp_path) -> None:
    original = artifact()
    destination = original.write(tmp_path / "ngram.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["forward"]["alpha"] = 9.0
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        BidirectionalNgramArtifact.read(destination)


def test_artifact_rejects_unknown_schema_fields() -> None:
    payload = artifact().as_dict()
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="keys mismatch"):
        BidirectionalNgramArtifact.from_dict(payload)
