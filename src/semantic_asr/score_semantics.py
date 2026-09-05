"""Compatibility import path for the canonical Semantic ASR score contract."""

from __future__ import annotations

from collections.abc import Mapping

from .score_contract import (
    CalibrationProfileRegistry,
    EvidenceScore,
    ScoreKind,
    ScoreMigrationError,
    ScoreNormalization,
    ScoreSemantics,
)


def probability_score(
    value: float,
    *,
    source: str,
    calibration_digest: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> EvidenceScore:
    """Construct a receipt-bearing legacy probability.

    The value still cannot be consumed as a correctness probability without a
    frozen :class:`CalibrationProfileRegistry` and its exact source score.
    """

    return EvidenceScore(
        value,
        kind=ScoreKind.PROBABILITY,
        source=source,
        calibrated=True,
        calibration_digest=calibration_digest,
        metadata=metadata,
    )


def uncalibrated_preference(
    value: float,
    *,
    source: str,
    metadata: Mapping[str, object] | None = None,
) -> EvidenceScore:
    return EvidenceScore(
        value,
        kind=ScoreKind.PREFERENCE,
        source=source,
        metadata=metadata,
    )


__all__ = [
    "CalibrationProfileRegistry",
    "EvidenceScore",
    "ScoreKind",
    "ScoreMigrationError",
    "ScoreNormalization",
    "ScoreSemantics",
    "probability_score",
    "uncalibrated_preference",
]
