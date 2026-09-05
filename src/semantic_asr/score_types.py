"""Canonical score types and deterministic held-out calibrators."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Protocol, Self

from .score_contract import (
    CalibrationProfile,
    CalibrationProfileRegistry,
    EvidenceScore,
    ScoreNormalization,
    ScoreProvenance,
    ScoreSemantics,
    require_sha256,
)

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class CalibrationExample:
    score: float
    correct: bool
    group: str | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not math.isfinite(float(self.score)):
            raise ValueError("calibration score must be finite and non-boolean")
        if (
            isinstance(self.weight, bool)
            or not math.isfinite(float(self.weight))
            or self.weight <= 0
        ):
            raise ValueError("calibration weight must be finite and positive")


class FittedCalibrator(Protocol):
    name: str

    def probability(
        self,
        score: EvidenceScore,
        *,
        profile: CalibrationProfile | None = None,
    ) -> EvidenceScore: ...

    def profile_for(
        self,
        score: EvidenceScore,
        *,
        dataset_split_digest: str | None = None,
    ) -> CalibrationProfile: ...

    @property
    def digest(self) -> str: ...


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-min(80.0, value))
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(max(-80.0, value))
    return exponent / (1.0 + exponent)


def _clip_probability(value: float) -> float:
    return min(1.0 - _EPS, max(_EPS, float(value)))


def _legacy_probability(
    *,
    source_score: EvidenceScore,
    value: float,
    calibration_digest: str,
    dataset_digest: str,
    calibrator: str,
) -> EvidenceScore:
    """Create a receipt-bearing legacy probability.

    It is intentionally not registry-applicable because no frozen applicability
    profile is attached. Decision code must call ``require_probability`` with a
    registry, which rejects this compatibility form.
    """

    provenance = ScoreProvenance(
        scorer=source_score.provenance.scorer,
        model=source_score.provenance.model,
        revision=source_score.provenance.revision,
        runtime=source_score.provenance.runtime,
        runtime_version=source_score.provenance.runtime_version,
        normalization=source_score.provenance.normalization,
        score_domain_digest=source_score.provenance.score_domain_digest,
        configuration_digest=source_score.provenance.configuration_digest,
        calibration_digest=calibration_digest,
        input_evidence_digest=source_score.provenance.input_evidence_digest,
        input_condition_digest=source_score.provenance.input_condition_digest,
        metadata={
            **source_score.provenance.metadata,
            "sourceSemantics": source_score.semantics.value,
            "sourceScoreDigest": source_score.digest,
            "calibrationDatasetDigest": dataset_digest,
            "calibrator": calibrator,
            "legacyUnregisteredCalibration": True,
        },
    )
    return EvidenceScore(
        value=value,
        semantics=ScoreSemantics.PROBABILITY,
        provenance=provenance,
        calibrated=True,
    )


def _profile_for_calibrator(
    *,
    score: EvidenceScore,
    calibration_digest: str,
    dataset_split_digest: str,
    method: str,
    name: str,
    source_semantics: ScoreSemantics,
    parameters: dict[str, object],
    parameters_digest: str,
) -> CalibrationProfile:
    require_sha256(dataset_split_digest, name="dataset_split_digest")
    if score.semantics != source_semantics:
        raise ValueError(f"calibrator expects {source_semantics}, got {score.semantics}")
    return CalibrationProfile(
        calibration_digest=calibration_digest,
        name=name,
        method=method,
        source_semantics=source_semantics,
        scorer=score.provenance.scorer,
        model=score.provenance.model,
        revision=score.provenance.revision,
        normalization=score.provenance.normalization,
        score_domain_digest=score.provenance.score_domain_digest,
        configuration_digest=score.provenance.configuration_digest,
        input_condition_digest=score.provenance.input_condition_digest,
        dataset_split_digest=dataset_split_digest,
        parameters=parameters,
        metadata={"parametersDigest": parameters_digest},
    )


@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    """A held-out logistic calibrator fitted without third-party dependencies."""

    slope: float
    intercept: float
    source_semantics: ScoreSemantics
    dataset_digest: str
    iterations: int
    l2: float
    name: str = "platt-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_semantics", ScoreSemantics(self.source_semantics))
        if self.source_semantics == ScoreSemantics.PROBABILITY:
            raise ValueError("source semantics should describe the uncalibrated input")
        if not all(
            math.isfinite(value)
            for value in (self.slope, self.intercept, float(self.iterations), self.l2)
        ):
            raise ValueError("calibrator parameters must be finite")
        if self.iterations < 1 or self.l2 < 0:
            raise ValueError("invalid calibrator settings")
        if not self.dataset_digest:
            raise ValueError("dataset digest is required")

    @classmethod
    def fit(
        cls,
        examples: list[CalibrationExample],
        *,
        source_semantics: ScoreSemantics,
        dataset_digest: str,
        learning_rate: float = 0.05,
        iterations: int = 2_000,
        l2: float = 1e-4,
    ) -> Self:
        if len(examples) < 4:
            raise ValueError("at least four held-out examples are required")
        if not any(example.correct for example in examples) or all(
            example.correct for example in examples
        ):
            raise ValueError("calibration examples must contain both outcomes")
        if learning_rate <= 0 or iterations < 1 or l2 < 0:
            raise ValueError("invalid optimization settings")

        total_weight = sum(example.weight for example in examples)
        positive = sum(example.weight for example in examples if example.correct)
        intercept = math.log(_clip_probability(positive / total_weight)) - math.log(
            _clip_probability(1.0 - positive / total_weight)
        )
        slope = 0.0

        mean = sum(example.weight * example.score for example in examples) / total_weight
        variance = (
            sum(example.weight * (example.score - mean) ** 2 for example in examples) / total_weight
        )
        scale = max(math.sqrt(variance), 1e-6)

        for step in range(iterations):
            grad_slope = 0.0
            grad_intercept = 0.0
            for example in examples:
                x = (example.score - mean) / scale
                prediction = _sigmoid(slope * x + intercept)
                error = prediction - float(example.correct)
                grad_slope += example.weight * error * x
                grad_intercept += example.weight * error
            grad_slope = grad_slope / total_weight + l2 * slope
            grad_intercept /= total_weight
            rate = learning_rate / math.sqrt(1.0 + step / 200.0)
            slope -= rate * grad_slope
            intercept -= rate * grad_intercept

        folded_slope = slope / scale
        folded_intercept = intercept - slope * mean / scale
        return cls(
            slope=folded_slope,
            intercept=folded_intercept,
            source_semantics=source_semantics,
            dataset_digest=dataset_digest,
            iterations=iterations,
            l2=l2,
        )

    def profile_for(
        self,
        score: EvidenceScore,
        *,
        dataset_split_digest: str | None = None,
    ) -> CalibrationProfile:
        return _profile_for_calibrator(
            score=score,
            calibration_digest=self.digest,
            dataset_split_digest=dataset_split_digest or self.dataset_digest,
            method="platt",
            name=self.name,
            source_semantics=self.source_semantics,
            parameters={"slope": self.slope, "intercept": self.intercept},
            parameters_digest=self.digest,
        )

    def probability(
        self,
        score: EvidenceScore,
        *,
        profile: CalibrationProfile | None = None,
    ) -> EvidenceScore:
        if score.semantics != self.source_semantics:
            raise ValueError(f"calibrator expects {self.source_semantics}, got {score.semantics}")
        value = _sigmoid(self.slope * score.value + self.intercept)
        if profile is None:
            return _legacy_probability(
                source_score=score,
                value=value,
                calibration_digest=self.digest,
                dataset_digest=self.dataset_digest,
                calibrator=self.name,
            )
        if profile.calibration_digest != self.digest:
            raise ValueError("calibration profile does not match this Platt artifact")
        return profile.probability(score, value)

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "name": self.name,
                "slope": self.slope,
                "intercept": self.intercept,
                "sourceSemantics": self.source_semantics.value,
                "datasetDigest": self.dataset_digest,
                "iterations": self.iterations,
                "l2": self.l2,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    """Monotone non-parametric calibration using pair-adjacent violators."""

    thresholds: tuple[float, ...]
    probabilities: tuple[float, ...]
    source_semantics: ScoreSemantics
    dataset_digest: str
    name: str = "isotonic-pav-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_semantics", ScoreSemantics(self.source_semantics))
        if not self.thresholds or len(self.thresholds) != len(self.probabilities):
            raise ValueError("thresholds and probabilities must have equal non-zero length")
        if tuple(sorted(self.thresholds)) != self.thresholds:
            raise ValueError("thresholds must be sorted")
        if any(not 0 <= value <= 1 for value in self.probabilities):
            raise ValueError("isotonic probabilities must be in [0, 1]")
        if any(
            left > right
            for left, right in zip(self.probabilities, self.probabilities[1:], strict=False)
        ):
            raise ValueError("isotonic probabilities must be monotone")
        if not self.dataset_digest:
            raise ValueError("dataset digest is required")

    @classmethod
    def fit(
        cls,
        examples: list[CalibrationExample],
        *,
        source_semantics: ScoreSemantics,
        dataset_digest: str,
    ) -> Self:
        if len(examples) < 4:
            raise ValueError("at least four held-out examples are required")
        ordered = sorted(examples, key=lambda example: (example.score, example.correct))
        blocks: list[list[float]] = []
        for example in ordered:
            if blocks and example.score == blocks[-1][1]:
                blocks[-1][2] += example.weight * float(example.correct)
                blocks[-1][3] += example.weight
            else:
                blocks.append(
                    [
                        float(example.score),
                        float(example.score),
                        example.weight * float(example.correct),
                        float(example.weight),
                    ]
                )
        index = 0
        while index < len(blocks) - 1:
            left = blocks[index][2] / blocks[index][3]
            right = blocks[index + 1][2] / blocks[index + 1][3]
            if left <= right:
                index += 1
                continue
            blocks[index] = [
                blocks[index][0],
                blocks[index + 1][1],
                blocks[index][2] + blocks[index + 1][2],
                blocks[index][3] + blocks[index + 1][3],
            ]
            del blocks[index + 1]
            index = max(0, index - 1)

        return cls(
            thresholds=tuple(block[1] for block in blocks),
            probabilities=tuple(block[2] / block[3] for block in blocks),
            source_semantics=source_semantics,
            dataset_digest=dataset_digest,
        )

    def profile_for(
        self,
        score: EvidenceScore,
        *,
        dataset_split_digest: str | None = None,
    ) -> CalibrationProfile:
        return _profile_for_calibrator(
            score=score,
            calibration_digest=self.digest,
            dataset_split_digest=dataset_split_digest or self.dataset_digest,
            method="isotonic-pav",
            name=self.name,
            source_semantics=self.source_semantics,
            parameters={
                "thresholds": self.thresholds,
                "probabilities": self.probabilities,
            },
            parameters_digest=self.digest,
        )

    def probability(
        self,
        score: EvidenceScore,
        *,
        profile: CalibrationProfile | None = None,
    ) -> EvidenceScore:
        if score.semantics != self.source_semantics:
            raise ValueError(f"calibrator expects {self.source_semantics}, got {score.semantics}")
        index = 0
        while index < len(self.thresholds) - 1 and score.value > self.thresholds[index]:
            index += 1
        value = self.probabilities[index]
        if profile is None:
            return _legacy_probability(
                source_score=score,
                value=value,
                calibration_digest=self.digest,
                dataset_digest=self.dataset_digest,
                calibrator=self.name,
            )
        if profile.calibration_digest != self.digest:
            raise ValueError("calibration profile does not match this isotonic artifact")
        return profile.probability(score, value)

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "name": self.name,
                "thresholds": self.thresholds,
                "probabilities": self.probabilities,
                "sourceSemantics": self.source_semantics.value,
                "datasetDigest": self.dataset_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def calibration_dataset_digest(examples: list[CalibrationExample]) -> str:
    payload = [
        {
            "score": example.score,
            "correct": example.correct,
            "group": example.group,
            "weight": example.weight,
        }
        for example in examples
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "CalibrationExample",
    "CalibrationProfile",
    "CalibrationProfileRegistry",
    "EvidenceScore",
    "FittedCalibrator",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "ScoreNormalization",
    "ScoreProvenance",
    "ScoreSemantics",
    "calibration_dataset_digest",
]
