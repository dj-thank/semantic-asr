"""Typed held-out-normalized evidence for multi-level ASR deliberation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .contracts import sha256_json
from .score_semantics import EvidenceScore, ScoreKind

UtilityChannel = Literal[
    "asr_acoustic",
    "phone",
    "mora",
    "discrete_unit",
    "lexical",
    "preservation",
    "semantic",
    "transition",
]
ArcOrigin = Literal[
    "first-pass",
    "phonetic-proposal",
    "context-proposal",
    "guarded-generation",
    "human",
]
ResolutionMode = Literal[
    "retained-first-pass",
    "context-resolved-orthography",
    "acoustic-context-consensus",
    "acoustically-verified-proposal",
]
DecisionStatus = Literal["accepted", "provisional"]

AUDIO_CHANNELS = frozenset({"asr_acoustic", "phone", "mora", "discrete_unit"})
INDEPENDENT_AUDIO_CHANNELS = frozenset({"phone", "mora", "discrete_unit"})
GENERATED_ORIGINS = frozenset({"phonetic-proposal", "context-proposal", "guarded-generation"})


def _strict_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _evidence_digest(score: EvidenceScore) -> str:
    return sha256_json(
        {
            "value": score.value,
            "kind": score.kind.value,
            "source": score.source,
            "calibrated": score.calibrated,
            "calibrationDigest": score.calibration_digest,
            "higherIsBetter": score.higher_is_better,
            "metadata": score.metadata,
        }
    )


@dataclass(frozen=True, slots=True)
class BoundedUtility:
    """A dimensionless held-out-normalized utility, not a probability."""

    channel: UtilityChannel
    value: float
    source: str
    profile_digest: str
    input_digest: str

    def __post_init__(self) -> None:
        value = _strict_float(self.value, name="bounded utility")
        if not -1.0 <= value <= 1.0:
            raise ValueError("bounded utility must be in [-1, 1]")
        if self.channel not in {
            "asr_acoustic",
            "phone",
            "mora",
            "discrete_unit",
            "lexical",
            "preservation",
            "semantic",
            "transition",
        }:
            raise ValueError("unknown utility channel")
        if not self.source:
            raise ValueError("bounded utility source is required")
        if not _is_sha256(self.profile_digest) or not _is_sha256(self.input_digest):
            raise ValueError("bounded utility digests must be SHA-256 values")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "channel": self.channel,
                "value": self.value,
                "source": self.source,
                "profileDigest": self.profile_digest,
                "inputDigest": self.input_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class UtilityCalibrationProfile:
    """Frozen affine+tanh mapping fitted on held-out data.

    This mapping only puts heterogeneous scores on a bounded utility scale. It does not turn a
    log likelihood, raw score or preference into a correctness probability.
    """

    channel: UtilityChannel
    score_source: str
    score_kind: ScoreKind
    center: float
    scale: float
    fitted_manifest_sha256: str
    revision: str
    higher_is_better: bool = True
    schema_version: str = "1"

    def __post_init__(self) -> None:
        center = _strict_float(self.center, name="calibration center")
        scale = _strict_float(self.scale, name="calibration scale")
        if scale <= 0:
            raise ValueError("calibration scale must be positive")
        if not self.score_source or not self.revision:
            raise ValueError("calibration source and revision are required")
        if not _is_sha256(self.fitted_manifest_sha256):
            raise ValueError("fitted_manifest_sha256 must be a SHA-256 value")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "channel": self.channel,
                "scoreSource": self.score_source,
                "scoreKind": self.score_kind.value,
                "center": self.center,
                "scale": self.scale,
                "fittedManifestSha256": self.fitted_manifest_sha256,
                "revision": self.revision,
                "higherIsBetter": self.higher_is_better,
                "transform": "tanh-affine-v1",
            }
        )

    def transform(self, score: EvidenceScore) -> BoundedUtility:
        if score.source != self.score_source:
            raise ValueError("score source does not match the frozen calibration profile")
        if score.kind != self.score_kind:
            raise ValueError("score kind does not match the frozen calibration profile")
        if score.higher_is_better != self.higher_is_better:
            raise ValueError("score direction does not match the frozen calibration profile")
        oriented = float(score.value) if self.higher_is_better else -float(score.value)
        center = self.center if self.higher_is_better else -self.center
        value = math.tanh((oriented - center) / self.scale)
        return BoundedUtility(
            channel=self.channel,
            value=value,
            source=f"{self.score_source}:{self.revision}",
            profile_digest=self.digest,
            input_digest=_evidence_digest(score),
        )
