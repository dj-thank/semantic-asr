"""Typed held-out-normalized evidence for multi-level ASR deliberation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from .contracts import sha256_json
from .score_contract import (
    EvidenceScore,
    ScoreNormalization,
    ScoreSemantics,
    require_sha256,
)

UtilityChannel = Literal[
    "first_pass",
    "asr_acoustic",
    "phone",
    "mora",
    "mora_shadow",
    "discrete_unit",
    "lexical",
    "preservation",
    "cross_model",
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

AUDIO_CHANNELS = frozenset({"asr_acoustic", "phone", "mora", "discrete_unit", "cross_model"})
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
    try:
        require_sha256(value, name="digest")
    except (TypeError, ValueError):
        return False
    return True


def _evidence_digest(score: EvidenceScore) -> str:
    return score.digest


@dataclass(frozen=True, slots=True)
class BoundedUtility:
    """Dimensionless held-out-normalized path utility, never a probability."""

    channel: UtilityChannel
    value: float
    source: str
    profile_digest: str
    input_digest: str
    factor_weight: float = 1.0

    def __post_init__(self) -> None:
        value = _strict_float(self.value, name="bounded utility")
        factor_weight = _strict_float(self.factor_weight, name="factor_weight")
        if not -1.0 <= value <= 1.0:
            raise ValueError("bounded utility must be in [-1, 1]")
        if not 0.0 <= factor_weight <= 1.0:
            raise ValueError("factor_weight must be in [0, 1]")
        if self.channel not in {
            "first_pass",
            "asr_acoustic",
            "phone",
            "mora",
            "mora_shadow",
            "discrete_unit",
            "lexical",
            "preservation",
            "cross_model",
            "semantic",
            "transition",
        }:
            raise ValueError("unknown utility channel")
        if not self.source:
            raise ValueError("bounded utility source is required")
        if not _is_sha256(self.profile_digest) or not _is_sha256(self.input_digest):
            raise ValueError("bounded utility digests must be SHA-256 values")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "factor_weight", factor_weight)

    @property
    def weighted_value(self) -> float:
        return self.value * self.factor_weight

    def with_factor_weight(self, value: float) -> BoundedUtility:
        return replace(self, factor_weight=value)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "channel": self.channel,
                "value": self.value,
                "source": self.source,
                "profileDigest": self.profile_digest,
                "inputDigest": self.input_digest,
                "factorWeight": self.factor_weight,
            }
        )


@dataclass(frozen=True, slots=True, init=False)
class UtilityCalibrationProfile:
    """Frozen affine+tanh mapping fitted on held-out data.

    This profile creates a bounded ranking utility, not a correctness probability.
    Both the score semantics and normalization must match.
    """

    channel: UtilityChannel
    score_source: str
    score_semantics: ScoreSemantics
    score_normalization: ScoreNormalization
    center: float
    scale: float
    fitted_manifest_sha256: str
    revision: str
    higher_is_better: bool
    schema_version: str

    def __init__(
        self,
        *,
        channel: UtilityChannel,
        score_source: str,
        score_semantics: ScoreSemantics | str | None = None,
        score_normalization: ScoreNormalization | str | None = None,
        center: float,
        scale: float,
        fitted_manifest_sha256: str,
        revision: str,
        higher_is_better: bool = True,
        schema_version: str = "2",
        score_kind: object | None = None,
    ) -> None:
        if score_semantics is not None and score_kind is not None:
            raise TypeError("pass score_semantics or legacy score_kind, not both")
        if score_kind is not None:
            raw_kind = str(getattr(score_kind, "value", score_kind))
            if raw_kind == "log_likelihood":
                if score_normalization is None:
                    if score_source.startswith(("ctc-phone:", "ctc-mora:")):
                        score_normalization = ScoreNormalization.MEAN_FRAME
                    else:
                        raise ValueError(
                            "legacy log_likelihood utility profiles require explicit normalization"
                        )
                normalization = ScoreNormalization(score_normalization)
                if normalization == ScoreNormalization.SEQUENCE:
                    semantics = ScoreSemantics.CUMULATIVE_LOG_LIKELIHOOD
                elif normalization in {
                    ScoreNormalization.MEAN_FRAME,
                    ScoreNormalization.MEAN_TOKEN,
                    ScoreNormalization.TOKEN_POWER,
                }:
                    semantics = ScoreSemantics.AVERAGE_LOG_LIKELIHOOD
                elif normalization == ScoreNormalization.PATH_NORMALIZED:
                    semantics = ScoreSemantics.LOG_PROBABILITY
                else:
                    raise ValueError("invalid log-likelihood normalization")
            else:
                mapping = {
                    "raw": ScoreSemantics.UNCALIBRATED_SCORE,
                    "logit": ScoreSemantics.LOGIT,
                    "preference": ScoreSemantics.PREFERENCE,
                }
                try:
                    semantics = mapping[raw_kind]
                except KeyError as exc:
                    raise ValueError(f"unsupported legacy score kind: {raw_kind!r}") from exc
                normalization = ScoreNormalization(
                    score_normalization or ScoreNormalization.NONE
                )
        else:
            if score_semantics is None:
                raise TypeError("score_semantics is required")
            semantics = ScoreSemantics(score_semantics)
            normalization = ScoreNormalization(
                score_normalization or ScoreNormalization.NONE
            )
        center_value = _strict_float(center, name="calibration center")
        scale_value = _strict_float(scale, name="calibration scale")
        if scale_value <= 0:
            raise ValueError("calibration scale must be positive")
        if not score_source or not revision:
            raise ValueError("calibration source and revision are required")
        require_sha256(fitted_manifest_sha256, name="fitted_manifest_sha256")
        if not isinstance(higher_is_better, bool):
            raise TypeError("higher_is_better must be bool")
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "score_source", score_source)
        object.__setattr__(self, "score_semantics", semantics)
        object.__setattr__(self, "score_normalization", normalization)
        object.__setattr__(self, "center", center_value)
        object.__setattr__(self, "scale", scale_value)
        object.__setattr__(self, "fitted_manifest_sha256", fitted_manifest_sha256)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "higher_is_better", higher_is_better)
        object.__setattr__(self, "schema_version", schema_version)

    @property
    def score_kind(self) -> str:
        """Legacy coarse view retained for serialized compatibility."""

        if self.score_semantics in {
            ScoreSemantics.CUMULATIVE_LOG_LIKELIHOOD,
            ScoreSemantics.AVERAGE_LOG_LIKELIHOOD,
            ScoreSemantics.LOG_PROBABILITY,
        }:
            return "log_likelihood"
        mapping = {
            ScoreSemantics.UNCALIBRATED_SCORE: "raw",
            ScoreSemantics.LOGIT: "logit",
            ScoreSemantics.PREFERENCE: "preference",
        }
        return mapping.get(self.score_semantics, self.score_semantics.value)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "channel": self.channel,
                "scoreSource": self.score_source,
                "scoreSemantics": self.score_semantics.value,
                "scoreNormalization": self.score_normalization.value,
                "center": self.center,
                "scale": self.scale,
                "fittedManifestSha256": self.fitted_manifest_sha256,
                "revision": self.revision,
                "higherIsBetter": self.higher_is_better,
                "transform": "tanh-affine-v1",
                "outputSemantics": ScoreSemantics.BOUNDED_UTILITY.value,
            }
        )

    def transform(
        self,
        score: EvidenceScore,
        *,
        factor_weight: float = 1.0,
    ) -> BoundedUtility:
        if score.provenance.scorer != self.score_source:
            raise ValueError("score source does not match the frozen calibration profile")
        if score.semantics != self.score_semantics:
            raise ValueError("score semantics do not match the frozen calibration profile")
        if score.provenance.normalization != self.score_normalization:
            raise ValueError("score normalization does not match the frozen calibration profile")
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
            factor_weight=factor_weight,
        )
