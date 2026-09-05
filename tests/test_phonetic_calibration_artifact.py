from __future__ import annotations

import json

import pytest

from semantic_asr.phonetic_runtime.calibration import (
    PhoneticCalibrationCandidate,
    PhoneticCalibrationExample,
    fit_ctc_utility_calibration,
)
from semantic_asr.phonetic_runtime.calibration_artifact import DualCTCUtilityArtifact
from semantic_asr.phonetic_evidence import PosteriorFrame, PosteriorSequence


def posterior(kind: str, symbols: tuple[str, ...]) -> PosteriorSequence:
    vocabulary = ("<blk>", *tuple(dict.fromkeys((*symbols, "x"))))
    sequence = ["<blk>"]
    for symbol in symbols:
        sequence.extend((symbol, "<blk>"))
    frames = tuple(
        PosteriorFrame.from_mapping(
            start_ms=index * 20,
            end_ms=(index + 1) * 20,
            probabilities={
                value: 0.9 if value == winner else 0.1 / (len(vocabulary) - 1)
                for value in vocabulary
            },
        )
        for index, winner in enumerate(sequence)
    )
    return PosteriorSequence(
        kind=kind,  # type: ignore[arg-type]
        blank_symbol="<blk>",
        vocabulary=vocabulary,
        frames=frames,
        encoder="fixture",
        encoder_revision="runtime-r1",
        label_set_revision=f"{kind}-r1",
        source_audio_sha256="a" * 64,
    )


def report(kind: str):
    correct = ("m", "a") if kind == "phone" else ("マ",)
    wrong = ("x", "a") if kind == "phone" else ("x",)
    example = PhoneticCalibrationExample(
        example_id=f"{kind}-example",
        posterior=posterior(kind, correct),
        candidates=(
            PhoneticCalibrationCandidate(
                candidate_id="correct",
                text="ま",
                symbols=correct,
                correct=True,
            ),
            PhoneticCalibrationCandidate(
                candidate_id="wrong",
                text="わ",
                symbols=wrong,
                correct=False,
            ),
        ),
    )
    return fit_ctc_utility_calibration(
        (example,),
        held_out_manifest_sha256="b" * 64,
        revision=f"{kind}-cal-r1",
    )


def artifact() -> DualCTCUtilityArtifact:
    return DualCTCUtilityArtifact.from_reports(
        report("phone"),
        report("mora"),
        name="fixture-utility",
        revision="r1",
        runtime_profile_digest="c" * 64,
    )


def test_utility_artifact_round_trip(tmp_path) -> None:
    original = artifact()
    destination = original.write(tmp_path / "utility.json")

    loaded = DualCTCUtilityArtifact.read(destination)

    assert loaded == original
    assert loaded.digest == original.digest


def test_utility_artifact_detects_tampering(tmp_path) -> None:
    destination = artifact().write(tmp_path / "utility.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["phonePairwiseAccuracy"] = 0.0
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        DualCTCUtilityArtifact.read(destination)
