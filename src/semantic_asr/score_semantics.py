"""Compatibility import path for the canonical Semantic ASR score contract.

New code should import from :mod:`semantic_asr.score_contract` or
:mod:`semantic_asr.score_types`. ``EvidenceScore`` is the exact same class from
the canonical module; this file no longer defines a competing numeric type.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Mapping

from .score_contract import (
    CalibrationProfileRegistry,
    EvidenceScore,
    ScoreMigrationError,
    ScoreNormalization,
    ScoreSemantics,
)


class ScoreKind(StrEnum):
    """Legacy coarse score names.

    ``LOG_LIKELIHOOD`` is intentionally insufficient by itself. Callers must
    declare whether the value is cumulative, mean-token, mean-frame, or
    path-normalized when constructing/migrating that legacy form.
    """

    RAW = "raw"
    PROBABILITY = "probability"
    LOGIT = "logit"
    LOG_LIKELIHOOD = "log_likelihood"
    PREFERENCE = "preference"


def probability_score(
    value: float,
    *,
    source: str,
    calibration_digest: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> EvidenceScore:
    """Construct a receipt-bearing legacy probability.

    The value still cannot be consumed as a correctness probability without a
    frozen :class:`CalibrationProfileRegistry`.
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
