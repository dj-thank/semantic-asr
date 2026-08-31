from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class ScoreKind(StrEnum):
    """Semantic meaning of a numerical score.

    Values intentionally describe *what a number means*, not which component
    produced it. Raw scores and preferences must be calibrated before they are
    presented as probabilities.
    """

    RAW = "raw"
    PROBABILITY = "probability"
    LOGIT = "logit"
    LOG_LIKELIHOOD = "log_likelihood"
    PREFERENCE = "preference"


@dataclass(frozen=True, slots=True)
class EvidenceScore:
    value: float
    kind: ScoreKind
    source: str
    calibrated: bool = False
    calibration_digest: str | None = None
    higher_is_better: bool = True
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("score source is required")
        if not math.isfinite(float(self.value)):
            raise ValueError("score value must be finite")
        if self.kind == ScoreKind.PROBABILITY and not 0.0 <= float(self.value) <= 1.0:
            raise ValueError("probability score must be in [0, 1]")
        if self.calibration_digest and not self.calibrated:
            raise ValueError("calibration digest requires calibrated=True")
        if self.calibrated and self.kind != ScoreKind.PROBABILITY:
            raise ValueError("only probability scores may be marked calibrated")

    @property
    def is_probability(self) -> bool:
        return self.kind == ScoreKind.PROBABILITY

    @property
    def usable_as_probability(self) -> bool:
        return self.is_probability and self.calibrated

    def require_probability(self) -> float:
        if not self.usable_as_probability:
            raise ValueError(
                f"{self.source} is {self.kind.value}, not a held-out calibrated probability"
            )
        return float(self.value)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> EvidenceScore:
        values = dict(row)
        values["kind"] = ScoreKind(values["kind"])
        if values.get("metadata") is not None:
            values["metadata"] = dict(values["metadata"])
        return cls(**values)


def probability_score(
    value: float,
    *,
    source: str,
    calibration_digest: str,
    metadata: dict[str, Any] | None = None,
) -> EvidenceScore:
    if not calibration_digest:
        raise ValueError("calibration_digest is required for probability_score")
    return EvidenceScore(
        value=float(value),
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
    metadata: dict[str, Any] | None = None,
) -> EvidenceScore:
    """Record a ranker's stated preference without pretending it is probability."""

    return EvidenceScore(
        value=float(value),
        kind=ScoreKind.PREFERENCE,
        source=source,
        calibrated=False,
        metadata=metadata,
    )
