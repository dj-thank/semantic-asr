from __future__ import annotations

import pytest

from semantic_asr.ranker_calibration import (
    RankerCalibrationProfile,
    RankerCalibrationSample,
    calibration_sample_from_row,
    fit_ranker_calibration,
)


def _samples() -> list[RankerCalibrationSample]:
    output: list[RankerCalibrationSample] = []
    for group_index in range(10):
        group_id = f"speaker-{group_index:02d}"
        output.extend(
            [
                RankerCalibrationSample(
                    sample_id=f"{group_id}-n0",
                    group_id=group_id,
                    score=0.00 + group_index * 0.002,
                    correct=False,
                ),
                RankerCalibrationSample(
                    sample_id=f"{group_id}-n1",
                    group_id=group_id,
                    score=0.40 + group_index * 0.002,
                    correct=False,
                ),
                RankerCalibrationSample(
                    sample_id=f"{group_id}-p0",
                    group_id=group_id,
                    score=1.60 + group_index * 0.002,
                    correct=True,
                ),
                RankerCalibrationSample(
                    sample_id=f"{group_id}-p1",
                    group_id=group_id,
                    score=2.00 + group_index * 0.002,
                    correct=True,
                ),
            ]
        )
    return output


def test_platt_calibration_improves_heldout_nll_and_stays_monotonic() -> None:
    result = fit_ranker_calibration(
        _samples(),
        name="fixture-calibration",
        source_ranker="fixture-ranker",
    )
    assert result.profile.sample_count == 40
    assert result.profile.group_count == 10
    assert result.profile.slope > 0
    assert result.after.negative_log_likelihood < result.before.negative_log_likelihood
    assert result.after.brier < result.before.brier
    low = result.profile.transform(0.0)
    high = result.profile.transform(2.0)
    assert low is not None and high is not None and high > low


def test_ranker_calibration_profile_roundtrip_preserves_digest() -> None:
    result = fit_ranker_calibration(
        _samples(),
        name="fixture-calibration",
        source_ranker="fixture-ranker",
    )
    restored = RankerCalibrationProfile.from_dict(result.profile.as_dict())
    assert restored.digest == result.profile.digest
    assert restored.transform(1.25) == pytest.approx(result.profile.transform(1.25))


def test_calibration_refuses_training_or_test_rows() -> None:
    with pytest.raises(ValueError, match="forbidden split"):
        calibration_sample_from_row(
            {
                "sampleId": "sample",
                "groupId": "speaker",
                "score": 1.0,
                "correct": True,
                "split": "test",
            },
            line_number=1,
        )


def test_calibration_requires_both_classes() -> None:
    samples = [
        RankerCalibrationSample(
            sample_id=f"sample-{index}",
            group_id=f"speaker-{index % 2}",
            score=float(index),
            correct=True,
        )
        for index in range(8)
    ]
    with pytest.raises(ValueError, match="both correct and incorrect"):
        fit_ranker_calibration(
            samples,
            name="invalid",
            source_ranker="fixture-ranker",
        )
