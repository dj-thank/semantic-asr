from __future__ import annotations

import pytest

from semantic_asr.phonetic_evidence import PosteriorFrame, PosteriorSequence
from semantic_asr.phonetic_runtime.calibration import (
    PhoneticCalibrationCandidate,
    PhoneticCalibrationExample,
    fit_ctc_utility_calibration,
)


def posterior(*, kind: str, winner: tuple[str, ...], revision: str) -> PosteriorSequence:
    vocabulary = ("<blk>", *tuple(dict.fromkeys(winner + ("x",))))
    frames = []
    sequence = ["<blk>"]
    for value in winner:
        sequence.extend((value, "<blk>"))
    for index, value in enumerate(sequence):
        high = 0.9
        rest = (1.0 - high) / (len(vocabulary) - 1)
        frames.append(
            PosteriorFrame.from_mapping(
                start_ms=index * 20,
                end_ms=(index + 1) * 20,
                probabilities={
                    symbol: high if symbol == value else rest for symbol in vocabulary
                },
            )
        )
    return PosteriorSequence(
        kind=kind,  # type: ignore[arg-type]
        blank_symbol="<blk>",
        vocabulary=vocabulary,
        frames=tuple(frames),
        encoder="fixture-encoder",
        encoder_revision=revision,
        label_set_revision=f"{kind}-labels-r1",
        source_audio_sha256="a" * 64,
    )


def example(index: int) -> PhoneticCalibrationExample:
    return PhoneticCalibrationExample(
        example_id=f"example-{index}",
        posterior=posterior(kind="phone", winner=("m", "a"), revision="model-r1"),
        candidates=(
            PhoneticCalibrationCandidate(
                candidate_id=f"correct-{index}",
                text="ま",
                symbols=("m", "a"),
                correct=True,
            ),
            PhoneticCalibrationCandidate(
                candidate_id=f"wrong-{index}",
                text="わ",
                symbols=("x", "a"),
                correct=False,
            ),
        ),
    )


def test_calibration_fits_bounded_utility_without_calling_it_probability() -> None:
    report = fit_ctc_utility_calibration(
        (example(1), example(2)),
        held_out_manifest_sha256="b" * 64,
        revision="phone-cal-r1",
    )

    assert report.channel == "phone"
    assert report.pairwise_accuracy == 1.0
    assert report.profile.channel == "phone"
    assert report.profile.score_source.startswith("ctc-phone:")
    assert not report.profile.digest == ""


def test_calibration_requires_exactly_one_correct_candidate() -> None:
    base = example(1)
    invalid = PhoneticCalibrationExample(
        example_id="invalid",
        posterior=base.posterior,
        candidates=tuple(
            PhoneticCalibrationCandidate(
                candidate_id=row.candidate_id,
                text=row.text,
                symbols=row.symbols,
                correct=False,
            )
            for row in base.candidates
        ),
    )

    with pytest.raises(ValueError, match="exactly one correct"):
        invalid


def test_calibration_rejects_mixed_model_score_sources() -> None:
    first = example(1)
    second = PhoneticCalibrationExample(
        example_id="example-2",
        posterior=posterior(kind="phone", winner=("m", "a"), revision="model-r2"),
        candidates=example(2).candidates,
    )

    with pytest.raises(ValueError, match="incompatible CTC score sources"):
        fit_ctc_utility_calibration(
            (first, second),
            held_out_manifest_sha256="b" * 64,
            revision="phone-cal-r1",
        )
